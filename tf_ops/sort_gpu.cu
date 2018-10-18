/////////////////////////////////////////////////////////////////////////////
/// \file sort_gpu.cu
///
/// \brief Cuda implementation of the operations to distribute a batch of 
///        point clouds into a set of uniform grids by using the radix sort 
///        algorithm, O(n).
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <cstdio>

#include "cuda_kernel_utils.h"

#define POINT_BLOCK_SIZE 128
#define OFFSET_BLOCK_SIZE 512

////////////////////////////////////////////////////////////////////////////////// GPU

/**
 *  Method to compute the key of each point.
 *  @param  pNumPoints      Number of points.
 *  @param  pBatchSize      Size of the batch.
 *  @param  pNumCells       Number of cells of the grid.
 *  @param  pAABBMinPoint   Minimum point of the grid (3 componenets).
 *  @param  pAABBMaxPoint   Maximum point of the grid (3 componenets).
 *  @param  pPoints         List of points.
 *  @param  pBatchIds       List of batch ids.
 *  @param  pOutKeys        Output parameter with the keys of each point.
 */
__global__ void calc_key(
    const int pNumPoints,
    const int pBatchSize,
    const int pNumCells,
    const float* __restrict__ pAABBMinPoint, 
    const float* __restrict__ pAABBMaxPoint, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    int* __restrict__ pOutKeys) 
{
    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(currentIndex < pNumPoints){
        int currBatchId = pBatchIds[currentIndex];
        int pointIndex = currentIndex * 3;

        float maxAabbSize = max(max(pAABBMaxPoint[currBatchId*3] - pAABBMinPoint[currBatchId*3], 
            pAABBMaxPoint[currBatchId*3+1] - pAABBMinPoint[currBatchId*3+1]), 
            pAABBMaxPoint[currBatchId*3+2] - pAABBMinPoint[currBatchId*3+2]);
        float cellSize = maxAabbSize/(float)pNumCells;

        int xCell = max(min((int)floor((pPoints[pointIndex] - pAABBMinPoint[currBatchId*3])/cellSize), pNumCells -1), 0);
        int yCell = max(min((int)floor((pPoints[pointIndex+1] - pAABBMinPoint[currBatchId*3+1])/cellSize), pNumCells -1), 0);
        int zCell = max(min((int)floor((pPoints[pointIndex+2] - pAABBMinPoint[currBatchId*3+2])/cellSize), pNumCells -1), 0);

        pOutKeys[currentIndex] = currBatchId*pNumCells*pNumCells*pNumCells + xCell*pNumCells*pNumCells + yCell*pNumCells + zCell;
    }
}

/**
 *  Method to update the counters of each cell.
 *  @param  pNumKeys        Number of keys.
 *  @param  pKeys           List of keys.
 *  @param  pOutCounters    Output parameter with the counters.
 */
__global__ void update_counters(
    const int pNumKeys,
    const int* __restrict__ pKeys, 
    int* __restrict__ pOutCounters) 
{
    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(currentIndex < pNumKeys)
        atomicAdd(&pOutCounters[pKeys[currentIndex]], 1);
}

/**
 *  Second method to finish to propagate the offsets.
 *  @param  pStep1          Boolean that indicates if this is the first step.
 *  @param  pNumOffsets     Number of offsets.
 *  @param  pNumOffsets2    Number of second level offsets.
 *  @param  pOffsets        Input/Output parameter with the list of offsets.
 *  @param  pNumOffsets2    Output parameter with the list of second level offsets.
 */
__global__ void propagate_offsets(
    const bool pStep1,
    const int pNumOffsets, 
    const int pNumOffsets2, 
    int* __restrict__ pOffsets, 
    int* __restrict__ pOffsets2) 
{
    __shared__ int groupCounter[OFFSET_BLOCK_SIZE];

	//Get the local and global counter.
	int currCounter = threadIdx.x;
	int currGlobalCounter = threadIdx.x + blockIdx.x * blockDim.x;

	//Update the shared memory.
	if(currGlobalCounter < pNumOffsets)
		groupCounter[currCounter] = pOffsets[currGlobalCounter];
	else
		groupCounter[currCounter] = 0;

	//SIMD scan.
	for(int i = 1; i <= OFFSET_BLOCK_SIZE/2; i*=2)
	{
		__syncthreads();

		//Get the values of the pass.
		int currIndex = currCounter + i;
		int value1 = 0;
		int value2 = 0;
		if(currIndex < OFFSET_BLOCK_SIZE){
			value1 = groupCounter[currCounter];
			value2 = groupCounter[currIndex];
		}

		__syncthreads();

		//Update with the new value.
		if(currIndex < OFFSET_BLOCK_SIZE)
			groupCounter[currIndex] = value1 + value2;
	}

	__syncthreads();

	//Save the counter into global memory.
	if(currGlobalCounter < pNumOffsets){
		if(currCounter > 0)
			pOffsets[currGlobalCounter] = groupCounter[currCounter-1];
		else
			pOffsets[currGlobalCounter] = 0;
	}

    if(pStep1){
        //Update the offset buffer.
        if(currCounter == (OFFSET_BLOCK_SIZE-1) && blockIdx.x < pNumOffsets2)
            pOffsets2[blockIdx.x] = groupCounter[OFFSET_BLOCK_SIZE-1];
    }else{
        //Update the second level offset buffer.
        if(currCounter > blockIdx.x && currCounter < pNumOffsets2)
            atomicAdd(&pOffsets2[currCounter], groupCounter[OFFSET_BLOCK_SIZE-1]);
    }
}

/**
 *  Method to determine the new indexs of the points.
 *  @param  pNumPoints      Number of points.
 *  @param  pKeys           Input parameter with the list of keys.
 *  @param  pCounters       Input/Output parameter with the list of counters.
 *  @param  pOffset         Input parameter with the list of first level offsets.
 *  @param  pOffset2        Input parameter with the list of second level offsets.
 *  @param  pOutNewIndexs   Output parameter with the list of new indexes.
 */
__global__ void determine_new_index(
    const int pNumPoints,
    const int* __restrict__ pKeys,
    int* __restrict__ pCounters,
    const int* __restrict__ pOffset,
    const int* __restrict__ pOffset2,
    int* __restrict__ pOutNewIndexs) 
{
    int currPointId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currPointId < pNumPoints){
        int counterIndex = pKeys[currPointId];
        int offsetIndex = counterIndex/OFFSET_BLOCK_SIZE;
        int globalOffsetIndex = offsetIndex/OFFSET_BLOCK_SIZE;
        int localIndex = atomicAdd(&pCounters[counterIndex], 1);
        int index = localIndex + pOffset[offsetIndex] + pOffset2[globalOffsetIndex];
        pOutNewIndexs[currPointId] = index;
    }
}


/**
 *  Method to move the points into their respective cells.
 *  @param  pNumPoints      Number of points.
 *  @param  pBatchSize      Size of the batch.
 *  @param  pNumFeatures    Number of features.
 *  @param  pPoints         Input parameter with the list of points.
 *  @param  pBatchIds       Input parameter with the list of batch ids.
 *  @param  pFeatures       Input parameter with the list of features.
 *  @param  pKeys           Input parameter with the list of keys.
 *  @param  pNewIndexs      Input parameter with the list of new indexs.
 *  @param  pOutPoints      Output parameter with the list of points.
 *  @param  pOutBatchIds    Output parameter with the list of batch ids.
 *  @param  pOutFeatures    Output parameter with the list of features.
 *  @param  pOutKeys        Output parameter with the list of keys.
 */
__global__ void move_points(
    const int pNumPoints,
    const int pBatchSize,
    const int pNumFeatures,
    const float* __restrict__ pPoints,
    const int* __restrict__ pBatchIds,
    const float* __restrict__ pFeatures,
    const int* __restrict__ pKeys,
    const int* __restrict__ pNewIndexs,
    float* __restrict__ pOutPoints,
    int* __restrict__ pOutBatchIds,
    float* __restrict__ pOutFeatures,
    int* __restrict__ pOutKeys) 
{
    int currPointId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currPointId < pNumPoints){
        int index = pNewIndexs[currPointId];
        pOutPoints[index*3] = pPoints[currPointId*3];
        pOutPoints[index*3 +1] = pPoints[currPointId*3 +1];
        pOutPoints[index*3 +2] = pPoints[currPointId*3 +2];
        for(int i = 0;  i < pNumFeatures; ++i)
            pOutFeatures[index*pNumFeatures + i] = pFeatures[currPointId*pNumFeatures + i];
        pOutKeys[index] = pKeys[currPointId];
        pOutBatchIds[index] = pBatchIds[currPointId];
    }
}

/**
 *  Method to update the indexs of the cells.
 *  @param  pNumPoints  Number of points.
 *  @param  pKeys       Input parameter with the points keys.
 *  @param  pIndexs     Output parameter with the pair of indexs of each cell in the grid.
 */
__global__ void save_indexs(
    const int pNumPoints,
    const int* __restrict__ pKeys,
    int* __restrict__ pIndexs) 
{
    int currIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(currIndex < pNumPoints){
        int currKey = pKeys[currIndex];
        int prevIndex = currIndex-1;
        int currKeyIndex = currKey*2;
        if(prevIndex < 0){
            pIndexs[currKeyIndex] = 0;
        }else if(currKey != pKeys[prevIndex]){
            pIndexs[currKeyIndex] = currIndex;
        }

        int nextIndex = currIndex+1;
        if(nextIndex >= pNumPoints){
            pIndexs[currKeyIndex+1] = pNumPoints;
        }else if(currKey != pKeys[nextIndex]){
            pIndexs[currKeyIndex+1] = nextIndex;
        }
    }
}

/**
 *  Method to update the indexs of the cells.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumFeatures            Number of features.
 *  @param  pOutputGrads            Gradients of the output of the operation.
 *  @param  pOututFeatureGrads      Gradients of the feature outputs of the operation.
 *  @param  pNewIndexs              New indexs of the points.
 *  @param  pOutInputGrads          Output parameter with the input gradients.
 *  @param  pOutInputFeatureGrads   Output parameter with the input gradients of the features.
 */
__global__ void compute_gradients(
    const int pNumPoints,
    const int pNumFeatures,
    const float* __restrict__ pOutputGrads,
    const float* __restrict__ pOututFeatureGrads,
    const int* __restrict__ pNewIndexs,
    float* __restrict__ pOutInputGrads,
    float* __restrict__ pOutInputFeatureGrads) 
{
    int currPointId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currPointId < pNumPoints){
        int index = pNewIndexs[currPointId];
        pOutInputGrads[currPointId*3] = pOutputGrads[index*3];
        pOutInputGrads[currPointId*3 +1] = pOutputGrads[index*3 +1];
        pOutInputGrads[currPointId*3 +2] = pOutputGrads[index*3 +2];
        for(int i = 0 ; i < pNumFeatures; ++i)
            pOutInputFeatureGrads[currPointId*pNumFeatures + i] = pOututFeatureGrads[index*pNumFeatures + i];
    }
}

/**
 *  Method to sort the features back.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumFeatures            Number of features.
 *  @param  pInFeatures             List of input features.
 *  @param  pNewIndexs              New indexs of the points.
 *  @param  pOutFeatures            Output parameter with the list of features.
 */
__global__ void sort_features_back(
    const int pNumPoints,
    const int pNumFeatures,
    const float* __restrict__ pInFeatures,
    const int* __restrict__ pNewIndexs,
    float* __restrict__ pOutFeatures) 
{
    int currPointId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currPointId < pNumPoints){
        int index = pNewIndexs[currPointId];
        for(int i = 0 ; i < pNumFeatures; ++i)
            pOutFeatures[currPointId*pNumFeatures + i] = pInFeatures[index*pNumFeatures + i];
    }
}

/**
 *  Method to sort the gradients of the features back.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumFeatures            Number of features.
 *  @param  pOutFeatureGrads        List of output gradients of the features.
 *  @param  pNewIndexs              New indexs of the points.
 *  @param  pInFeatureGrad          Output parameter with the list of input gradients of the features.
 */
__global__ void sort_features_back_grad(
    const int pNumPoints,
    const int pNumFeatures,
    const float* __restrict__ pOutFeatureGrads,
    const int* __restrict__ pNewIndexs,
    float* __restrict__ pInFeatureGrad) 
{
    int currPointId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currPointId < pNumPoints){
        int index = pNewIndexs[currPointId];
        for(int i = 0 ; i < pNumFeatures; ++i)
            pInFeatureGrad[index*pNumFeatures + i] = pOutFeatureGrads[currPointId*pNumFeatures + i];
    }
}

/**
 *  Method to compute the inverse of the new position index list.
 *  @param  pNumPoints              Number of points.
 *  @param  pIndexs`                List of the new position of each point.
 *  @param  pOutIndexs              Output parameter with the list of old positions for each sorted point.
 */
__global__ void compute_inverse_indexs(
    const int pNumPoints,
    const int* __restrict__ pIndexs,
    int* __restrict__ pOutIndexs)
{
    int currPointId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currPointId < pNumPoints){
        int newIndex = pIndexs[currPointId];
        pOutIndexs[newIndex] = currPointId;
    }
}

/**
 *  Method to transofrm a list of indexs.
 *  @param  pNumIndexs              Number of indexs to transform.
 *  @param  pStartIndexs            List of indexs to transform.
 *  @param  pNewIndexs              List of the new position of each point.
 *  @param  pOutIndexs              Output parameter with the list of transformed indexs.
 */
__global__ void transform_indexs(
    const int pNumIndexs,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNewIndexs,
    int* __restrict__ pOutIndexs) 
{
    int currIndexId = threadIdx.x + blockIdx.x * blockDim.x;
    if(currIndexId < pNumIndexs){
        int index = pStartIndexs[currIndexId];
        int newIndex = pNewIndexs[index];
        pOutIndexs[currIndexId] = newIndex;
    }
}


/**
 *  Method to determine the cell size.
 *  @param  pBatchSize              Number of elements per batch.
 *  @param  pCellSize               Desired cell size.
 *  @param  pAABBMin                Minimum points of the bounding boxes.
 *  @param  pAABBMax                Maximum points of the bounding boxes.
 *  @param  pNumCells               Output parameter with the number of cells.
 */
__global__ void determine_cell_size(
    const int pBatchSize,
    const float pCellSize,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    int* __restrict__ pNumCells) 
{
    int currBatchId = threadIdx.x;
    if(currBatchId == 0){
        float maxAabbSize = max(max(pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
        int numCells = (int)(maxAabbSize/pCellSize);
        numCells = (numCells == 0)?1:numCells;
        //printf("Num cells: %d %f\n", numCells, maxAabbSize);
        *pNumCells = numCells;
    }
}



////////////////////////////////////////////////////////////////////////////////// CPU

int determineNumCells(
    const bool pScaleInv,
    const int pBatchSize,
    const float pCellSize,
    const float* pAABBMin, 
    const float* pAABBMax)
{
    if(pScaleInv){
        int numCellsCPU = (int)(1.0f/pCellSize);
        numCellsCPU = (numCellsCPU == 0)?1:numCellsCPU;
        return numCellsCPU;
    }

    int* numCells;
    cudaMalloc(&numCells, sizeof(int));
    cudaMemset(numCells, 0x3F, sizeof(int));

    determine_cell_size<<<1, pBatchSize>>>(pBatchSize, pCellSize, pAABBMin, pAABBMax, numCells);

    int numCellsCPU = 0;
    cudaMemcpy(&numCellsCPU, numCells, sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaFree(numCells));
    return numCellsCPU;
}

void computeAuxiliarBuffersSize(
    const int pBatchSize, 
    const int pNumCells,
    int* PBufferSize1,
    int* PBufferSize2,
    int* PBufferSize3)
{
    (*PBufferSize1) = pBatchSize*pNumCells*pNumCells*pNumCells;
    (*PBufferSize2) = (*PBufferSize1)/OFFSET_BLOCK_SIZE;
    (*PBufferSize2) += (((*PBufferSize1)%OFFSET_BLOCK_SIZE) != 0)?1:0;
    (*PBufferSize3) = (*PBufferSize2)/OFFSET_BLOCK_SIZE;
    (*PBufferSize3) += (((*PBufferSize2)%OFFSET_BLOCK_SIZE) != 0)?1:0;
}

void sortPointsStep1GPUKernel(
    const int pNumPoints, 
    const int pBatchSize,
    const int pNumCells,
    const float* pAABBMin, 
    const float* pAABBMax, 
    const float* pPoints,
    const int* pBatchIds,
    int* pAuxBuffCounters,
    int* pAuxBuffOffsets,
    int* pAuxBuffOffsets2,
    int* pKeys,
    int* pNewIndexs)
{
    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;
    int totalNumCells = pBatchSize*pNumCells*pNumCells*pNumCells;
    
    cudaMemset(pAuxBuffCounters, 0, totalNumCells*sizeof(int));

    int numOffsets = totalNumCells/OFFSET_BLOCK_SIZE;
    numOffsets += ((totalNumCells%OFFSET_BLOCK_SIZE) != 0)?1:0;
    int numOffsets2 = numOffsets/OFFSET_BLOCK_SIZE;
    numOffsets2 += ((numOffsets%OFFSET_BLOCK_SIZE) != 0)?1:0;

    cudaMemset(pAuxBuffOffsets, 0, numOffsets*sizeof(int));
    cudaMemset(pAuxBuffOffsets2, 0, numOffsets2*sizeof(int));

    calc_key<<<numBlocksPoints,POINT_BLOCK_SIZE>>>(
        pNumPoints, pBatchSize, pNumCells, pAABBMin, pAABBMax, pPoints, pBatchIds, pKeys);
    update_counters<<<numBlocksPoints,POINT_BLOCK_SIZE>>>(pNumPoints, pKeys, pAuxBuffCounters);
    propagate_offsets<<<numOffsets, OFFSET_BLOCK_SIZE>>>(true, totalNumCells, numOffsets, pAuxBuffCounters, pAuxBuffOffsets);
    propagate_offsets<<<numOffsets2, OFFSET_BLOCK_SIZE>>>(false, numOffsets, numOffsets2, pAuxBuffOffsets, pAuxBuffOffsets2);
    determine_new_index<<<numBlocksPoints,POINT_BLOCK_SIZE>>>(pNumPoints, pKeys, pAuxBuffCounters, pAuxBuffOffsets, 
        pAuxBuffOffsets2, pNewIndexs);
}

void sortPointsStep2GPUKernel(
    const int pNumPoints, 
    const int pBatchSize,
    const int pNumFeatures, 
    const int pNumCells,
    const float* pPoints,
    const int* pBatchIds,
    const float* pFeatures,
    const int* pKeys,
    const int* pNewIndexs,
    int* pAuxBuffer,
    float* pOutPoints,
    int* pOutBatchIds,
    float* pOutFeatures,
    int* pOutCellIndexs)
{    
    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;

    cudaMemset(pOutCellIndexs, 0, pBatchSize*pNumCells*pNumCells*pNumCells*sizeof(int)*2);

    move_points<<<numBlocksPoints,POINT_BLOCK_SIZE>>>
        (pNumPoints, pBatchSize, pNumFeatures, pPoints, pBatchIds, pFeatures, pKeys, pNewIndexs, pOutPoints, pOutBatchIds, pOutFeatures, pAuxBuffer);
    save_indexs<<<numBlocksPoints,POINT_BLOCK_SIZE>>>(pNumPoints, pAuxBuffer, pOutCellIndexs);
}

void sortPointsStep2GradGPUKernel(
    const int pNumPoints, 
    const int pNumFeatures,
    const float* pOutGradients,
    const float* pOutFeatureGradients,
    const int* pNewIndexs,
    float* pInGradients, 
    float* pInFeatureGradients)
{
    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;

    compute_gradients<<<numBlocksPoints,POINT_BLOCK_SIZE>>>
        (pNumPoints, pNumFeatures, pOutGradients, pOutFeatureGradients, pNewIndexs, pInGradients, pInFeatureGradients);
}


void sortFeaturesBack(
    const int pNumPoints,
    const int pNumFeatures,
    const float* pInFeatures,
    const int* pIndexs,
    float* pOutFeatures)
{
     int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;

    sort_features_back<<<numBlocksPoints,POINT_BLOCK_SIZE>>> (pNumPoints, pNumFeatures, pInFeatures, pIndexs, pOutFeatures);
}

void sortFeaturesBackGrad(
    const int pNumPoints,
    const int pNumFeatures,
    const float* pOutFeatureGrads,
    const int* pIndexs,
    float* pInFeatureGrads)
{
    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;

    sort_features_back_grad<<<numBlocksPoints,POINT_BLOCK_SIZE>>> (pNumPoints, pNumFeatures, pOutFeatureGrads, pIndexs, pInFeatureGrads);
}

void computeInverseIndexs(
    const int pNumPoints,
    const int* pIndexs,
    int* pOutIndexs)
{
    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;

    compute_inverse_indexs<<<numBlocksPoints,POINT_BLOCK_SIZE>>> (pNumPoints, pIndexs, pOutIndexs);
}

void transformIndexs(
    const int pNumIndexs, 
    const int pNumPoints, 
    const int* pInStartIndexs, 
    const int* pInNewIndexs, 
    int* pOutIndexs)
{
    int numBlocksPoints = pNumIndexs/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumIndexs%POINT_BLOCK_SIZE != 0)?1:0;

    transform_indexs<<<numBlocksPoints, POINT_BLOCK_SIZE>>>(pNumIndexs, pInStartIndexs, pInNewIndexs, pOutIndexs);
}
