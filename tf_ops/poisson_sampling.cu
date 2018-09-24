/////////////////////////////////////////////////////////////////////////////
/// \file poisson_sampling.cu
///
/// \brief Cuda implementation of the operations to perform a poisson disk 
///        sampling on a batch of point clouds (O(n)), to obtain the  
///        associated features to  the selected points, and to propagate the 
///        feature gradients.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <float.h>

#include "cuda_kernel_utils.h"

#define BLOCK_SIZE 4
#define PT_BLOCK_SIZE 128

////////////////////////////////////////////////////////////////////////////////// GPU

__constant__ int cellOffsetsPool[27][3];

/**
 *  Method to select a set of points from a point cloud in which all of them are at 
 *  distance [pRadius*0.5, pRadius].
 *  @param  scaleInv                Scale invariant.
 *  @param  pCurrBatch              Current batch processed.
 *  @param  pCurrentCell            Integer with the current cell of the block.
 *  @param  pNumPoints              Number of points.
 *  @param  pBatchSize              Size of the batch.
 *  @param  pNumCells               Number of cells of the grid.
 *  @param  pRadius                 Radius of the possion disk.
 *  @param  pAABBMinPoint           Minimum point of the grid (3 componenets).
 *  @param  pAABBMaxPoint           Maximum point of the grid (3 componenets).
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of the batch identifies.
 *  @param  pPDFs                   List of pdfs of each point.
 *  @param  pCellIndexs             Indexs of the grid cells.
 *  @param  pAuxBoleanBuffer        Input/Output parameter with the list of booleans indicating
 *      if a point was selected.
 *  @param  pOutSampledPoints       Output parameter with the list of sampled points.
 *  @param  pOutSampleBatchIds      Output parameter with the list of sampled batch ids.
 *  @param  pOutSampleIndexs        Output parameter with the list of indexs of the sampled points.
 *  @param  pOutNumSelectedPoints   Output parameter with the number of selected points.
 */
__global__ void selectSamples(
    const bool scaleInv, 
    const int pCurrBatch,
    const int pCurrentCell,
    const int pNumPoints,
    const int pBatchSize,
    const int pNumCells,
    const float pRadius,
    const float* __restrict__ pAABBMinPoint, 
    const float* __restrict__ pAABBMaxPoint, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const int* __restrict__ pCellIndexs,
    bool* __restrict__ pAuxBooleanBuffer,
    float* __restrict__ pOutSampledPoints,
    int* __restrict__ pOutSampleBatchIds,
    int* __restrict__ pOutSampleIndexs,
    int* __restrict__ pOutNumSelectedPoints) 
{
    int xCell = (threadIdx.x + blockIdx.x * blockDim.x)*3 + 1 + cellOffsetsPool[pCurrentCell][0];
    int yCell = (threadIdx.y + blockIdx.y * blockDim.y)*3 + 1 + cellOffsetsPool[pCurrentCell][1];
    int zCell = (threadIdx.z + blockIdx.z * blockDim.z)*3 + 1 + cellOffsetsPool[pCurrentCell][2];
    
    if(xCell < pNumCells && yCell < pNumCells & zCell < pNumCells){

        float maxAabbSize = max(max(
            pAABBMaxPoint[pCurrBatch*3] - pAABBMinPoint[pCurrBatch*3], 
            pAABBMaxPoint[pCurrBatch*3 + 1] - pAABBMinPoint[pCurrBatch*3 + 1]), 
            pAABBMaxPoint[pCurrBatch*3 + 2] - pAABBMinPoint[pCurrBatch*3 + 2]);
        float radius = (scaleInv)?pRadius*maxAabbSize:pRadius;

        int cellIndex = pCurrBatch*pNumCells*pNumCells*pNumCells + xCell*pNumCells*pNumCells + yCell*pNumCells + zCell;
        int initPoint = pCellIndexs[cellIndex*2];
        int endPoint = pCellIndexs[cellIndex*2 +1];
        for(int i = initPoint; i < endPoint; ++i)
        {
            float centralCoords[3] = {pPoints[i*3], pPoints[i*3+1], pPoints[i*3+2]};
            bool collision = false;
            
            for(int neighIter = 0; (neighIter < 27) && !collision; ++neighIter)
            {
                int currCellIndex[3] = {xCell+cellOffsetsPool[neighIter][0], yCell+cellOffsetsPool[neighIter][1], zCell+cellOffsetsPool[neighIter][2]};
                if(currCellIndex[0] >= 0 && currCellIndex[0] < pNumCells &&
                    currCellIndex[1] >= 0 && currCellIndex[1] < pNumCells &&
                    currCellIndex[2] >= 0 && currCellIndex[2] < pNumCells)
                {
                    int cellIndexFlat = pCurrBatch*pNumCells*pNumCells*pNumCells + currCellIndex[0]*pNumCells*pNumCells + currCellIndex[1]*pNumCells + currCellIndex[2];
                    int initNeighIndex = pCellIndexs[cellIndexFlat*2];
                    int endNeighIndex = pCellIndexs[cellIndexFlat*2 + 1];
                    for(int j = initNeighIndex; (j < endNeighIndex) && !collision; ++j)
                    {
                        int currPointIndex = j * 3;
                        float currentCoords[3] = {pPoints[currPointIndex], pPoints[currPointIndex+1], pPoints[currPointIndex+2]};
                        float diffVector[3] = {currentCoords[0] - centralCoords[0], currentCoords[1] - centralCoords[1], currentCoords[2] - centralCoords[2]};
                        float pointDist = sqrt(diffVector[0]*diffVector[0] + diffVector[1]*diffVector[1] + diffVector[2]*diffVector[2]);
                        if(pointDist < radius && pAuxBooleanBuffer[j]){
                            collision = true;
                        }
                    }
                }
            }

            if(!collision){
                pAuxBooleanBuffer[i] = true;
                int finalPointIndex = atomicAdd(&pOutNumSelectedPoints[0], 1);
                pOutSampledPoints[finalPointIndex*3] = centralCoords[0];
                pOutSampledPoints[finalPointIndex*3+1] = centralCoords[1];
                pOutSampledPoints[finalPointIndex*3+2] = centralCoords[2];
                pOutSampleBatchIds[finalPointIndex] = pCurrBatch;
                pOutSampleIndexs[finalPointIndex] = i;
            }
        }   
    }
}


/**
 *  Method to get the features of the sampled points.
 *  @param  pNumSamples         Number of samples.
 *  @param  pNumFeatures        Number of features.
 *  @param  pSampledIndexs      List of indexs of the sampled points.
 *  @param  pFeatures           List of input features.
 *  @param  pOutSampledFeatures List of output sampled features.
 */
__global__ void selectFeatureSamples(
    const int pNumSamples,
    const int pNumFeatures,
    const int* __restrict__ pSampledIndexs, 
    const float* __restrict__ pFeatures,
    float* __restrict__ pOutSampledFeatures) 
{
    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int sampleIndex = currentIndex/pNumFeatures;
    int featureIndex = currentIndex%pNumFeatures;
    if(sampleIndex < pNumSamples){
        pOutSampledFeatures[currentIndex] = pFeatures[pSampledIndexs[sampleIndex]*pNumFeatures + featureIndex];
    }
}

/**
 *  Method to get the gradients of the features of the sampled points.
 *  @param  pNumSamples                 Number of samples.
 *  @param  pNumFeatures                Number of features.
 *  @param  pSampledIndexs              List of indexs of the sampled points.
 *  @param  pFeaturesGrads              List of gradients of output features.
 *  @param  pOutSampledFeaturesGrads    List of output gradients of input features.
 */
__global__ void selectFeatureSamplesGrad(
    const int pNumSamples,
    const int pNumFeatures,
    const int* __restrict__ pSampledIndexs, 
    const float* __restrict__ pFeaturesGrads,
    float* __restrict__ pOutSampledFeaturesGrads) 
{
    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int sampleIndex = currentIndex/pNumFeatures;
    int featureIndex = currentIndex%pNumFeatures;
    if(sampleIndex < pNumSamples){
        pOutSampledFeaturesGrads[pSampledIndexs[sampleIndex]*pNumFeatures + featureIndex] = pFeaturesGrads[currentIndex];
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

int samplePointCloud(
    const bool scaleInv, 
    const float pRadius,
    const int pNumPoints,
    const int pBatchSize,
    const int pNumCells,
    const float* pAABBMin,
    const float* pAABBMax,
    const float* pPoints,
    const int* pBatchIds,
    const int* pCellIndexs,
    float* pSelectedPts,
    int* pSelectedBatchIds,
    int* pSelectedIndexs,
    bool* pAuxBoolBuffer)
{
    //Init device symbols.
    int cellOffsetsPoolCPU[27][3] = {
        {1, 1, -1}, {0, -1, 1}, {0, 1, 1}, {0, 1, 0}, {0, 0, 1}, {0, -1, 0}, {-1, 1, -1},
        {0, -1, -1}, {1, 0, 0}, {1, -1, 1}, {1, 0, 1}, {-1, 1, 1}, {-1, 0, 0}, {1, -1, -1},
        {0, 1, -1}, {-1, -1, 0}, {-1, 1, 0}, {0, 0, 0}, {0, 0, -1}, {1, 1, 0}, {1, 0, -1},
        {1, -1, 0}, {-1, 0, 1}, {1, 1, 1}, {-1, 0, -1}, {-1, -1, -1}, {-1, -1, 1}};
    cudaMemcpyToSymbol(cellOffsetsPool, cellOffsetsPoolCPU, 27*3*sizeof(int));
    int numSelectedPointsCPU = 0;

    gpuErrchk(cudaMemset(pAuxBoolBuffer, 0, sizeof(bool)*pNumPoints));

    int* numSelectedPoints;
    gpuErrchk(cudaMalloc(&numSelectedPoints, sizeof(int)));
    gpuErrchk(cudaMemset(numSelectedPoints, 0, sizeof(int)));

    int numPhaseGroups = pNumCells/3;
    numPhaseGroups += (pNumCells%3!=0)?1:0;
    int numBlocks = numPhaseGroups/BLOCK_SIZE;
    numBlocks += (numPhaseGroups%BLOCK_SIZE!=0)?1:0;
    for(int b = 0; b < pBatchSize; ++b){
        for(int i = 0; i < 27; ++i){
            selectSamples<<<dim3(numBlocks,numBlocks,numBlocks), dim3(BLOCK_SIZE,BLOCK_SIZE,BLOCK_SIZE)>>>
                (scaleInv, b, i, pNumPoints, pBatchSize, pNumCells, pRadius, pAABBMin, 
                pAABBMax, pPoints, pBatchIds, pCellIndexs, pAuxBoolBuffer, pSelectedPts, 
                pSelectedBatchIds, pSelectedIndexs, numSelectedPoints);

            gpuErrchk(cudaPeekAtLastError());
        }
    }

    //Copy from GPU the number of selected samples.
    gpuErrchk(cudaMemcpy(&numSelectedPointsCPU, numSelectedPoints, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(numSelectedPoints));

#ifdef PRINT_CONV_INFO
    printf("Num Cells: %d | Input points: %d | Result pooling: %d\n", pNumCells, pNumPoints, numSelectedPointsCPU);
#endif

    return numSelectedPointsCPU;
}

void copyPoints(
    float* pSelectedPts,
    int* pSelectedBatchIds,
    int* pSelectedIndexs,
    const int pNumPts,
    float* pDestPts,
    int* pDestBatchIds,
    int* pDestIndexs)
{
    gpuErrchk(cudaMemcpy(pDestPts, pSelectedPts, sizeof(float)*3*pNumPts, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(pDestBatchIds, pSelectedBatchIds, sizeof(int)*pNumPts, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(pDestIndexs, pSelectedIndexs, sizeof(int)*pNumPts, cudaMemcpyDeviceToDevice));
}

void getFeaturesSampledPoints(
    int pNumPoints, 
    int pNumFeatures, 
    int pNumSampledPoints, 
    const int* pInPointsIndexs, 
    const float* pInFeature, 
    float* pOutSelFeatures)
{
    int numBlocksPoints = pNumSampledPoints/PT_BLOCK_SIZE;
    numBlocksPoints += (pNumSampledPoints%PT_BLOCK_SIZE != 0)?1:0;
    selectFeatureSamples<<<pNumSampledPoints, PT_BLOCK_SIZE>>>(pNumSampledPoints, pNumFeatures, pInPointsIndexs, pInFeature, pOutSelFeatures);
    gpuErrchk(cudaPeekAtLastError());
}

void getFeaturesSampledPointsGradients(
    int pNumPoints, 
    int pNumFeatures, 
    int pNumSampledPoints, 
    const int* pInPointsIndexs, 
    const float* pInOutFeatureGrad, 
    float* pOutInFeaturesGradients)
{
     gpuErrchk(cudaMemset(pOutInFeaturesGradients, 0, sizeof(int)*pNumFeatures*pNumPoints));
     
    int numBlocksPoints = pNumSampledPoints/PT_BLOCK_SIZE;
    numBlocksPoints += (pNumSampledPoints%PT_BLOCK_SIZE != 0)?1:0;
    selectFeatureSamplesGrad<<<pNumSampledPoints, PT_BLOCK_SIZE>>>(pNumSampledPoints, pNumFeatures, pInPointsIndexs, pInOutFeatureGrad, pOutInFeaturesGradients);
    gpuErrchk(cudaPeekAtLastError());
}
