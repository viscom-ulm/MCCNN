/////////////////////////////////////////////////////////////////////////////
/// \file find_neighbors.cu
///
/// \brief Cuda implementation of the operation to find the neighboring 
///        points within a certain radius.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <iostream>
#include <fstream>

#include "cuda_kernel_utils.h"

#define POINT_BLOCK_SIZE 128
#define POINT_BLOCK_PACK_SIZE 256

////////////////////////////////////////////////////////////////////////////////// GPU

__constant__ int cellOffsets[27][3];

/**
 *  Method to count the neighboring points for each point.
 *  @param  pNumPoints          Number of points.
 *  @param  pNumCells           Number of cells of the grid.
 *  @param  pAABBMinPoint       Minimum point of the grid (3 componenets).
 *  @param  pAABBMaxPoint       Maximum point of the grid (3 componenets).
 *  @param  pPoints             List of points.
 *  @param  pBatchIds           List of batch ids.
 *  @param  pPoints2            List of points from where to find neighbors.
 *  @param  pCellIndexs         Indexs of the grid cells.
 *  @param  pOutNeigbors        Output parameter with the number of neighbors of each point.
 *  @param  pOutNumNeigbors     Output parameter with the total number of neighbors.
 */
__global__ void countNeighbors(
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumCells,
    const float pRadius,
    const float* __restrict__ pAABBMinPoint, 
    const float* __restrict__ pAABBMaxPoint, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const float* __restrict__ pPoints2, 
    const int* __restrict__ pCellIndexs,
    int* __restrict__ pOutNeigbors,
    int* __restrict__ pOutNumNeigbors) 
{
    __shared__ int blockTotalNeighbors;

    if(threadIdx.x == 0){
        blockTotalNeighbors = 0;
    }

    __syncthreads();

    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(currentIndex < pNumPoints){
        int currBatchId = pBatchIds[currentIndex];
        int pointIndex = currentIndex * 3;
        
        float maxAabbSize = max(max(
            pAABBMaxPoint[currBatchId*3] - pAABBMinPoint[currBatchId*3], 
            pAABBMaxPoint[currBatchId*3+1] - pAABBMinPoint[currBatchId*3+1]), 
            pAABBMaxPoint[currBatchId*3+2] - pAABBMinPoint[currBatchId*3+2]);
        float cellSize = maxAabbSize/(float)pNumCells;
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;
        
        float centralCoords[3] = {pPoints[pointIndex], pPoints[pointIndex+1], pPoints[pointIndex+2]};
        int xCell = max(min((int)floor((centralCoords[0] - pAABBMinPoint[currBatchId*3])/cellSize), pNumCells -1), 0);
        int yCell = max(min((int)floor((centralCoords[1] - pAABBMinPoint[currBatchId*3+1])/cellSize), pNumCells -1), 0);
        int zCell = max(min((int)floor((centralCoords[2] - pAABBMinPoint[currBatchId*3+2])/cellSize), pNumCells -1), 0);

        int neighborIter = 0;
        for(int i = 0; i < 27; ++i)
        {
            int currCellIndex[3] = {xCell+cellOffsets[i][0], yCell+cellOffsets[i][1], zCell+cellOffsets[i][2]};
            if(currCellIndex[0] >= 0 && currCellIndex[0] < pNumCells &&
                currCellIndex[1] >= 0 && currCellIndex[1] < pNumCells &&
                currCellIndex[2] >= 0 && currCellIndex[2] < pNumCells)
            {
                int cellIndexFlat = currBatchId*pNumCells*pNumCells*pNumCells + currCellIndex[0]*pNumCells*pNumCells + currCellIndex[1]*pNumCells + currCellIndex[2];
                int initIndex = pCellIndexs[cellIndexFlat*2];
                int endIndex = pCellIndexs[cellIndexFlat*2 + 1];
                
                for(int j = initIndex; j < endIndex; ++j)
                {
                    int currPointIndex = j * 3;
                    float currentCoords[3] = {pPoints2[currPointIndex], pPoints2[currPointIndex+1], pPoints2[currPointIndex+2]};
                    float diffVector[3] = {currentCoords[0] - centralCoords[0], currentCoords[1] - centralCoords[1], currentCoords[2] - centralCoords[2]};
                    float pointDist = sqrt(diffVector[0]*diffVector[0] + diffVector[1]*diffVector[1] + diffVector[2]*diffVector[2]);
                    if(pointDist < scaledRadius){
                        neighborIter++;
                    }
                }
            }
        }

        pOutNeigbors[currentIndex] = neighborIter;
        atomicAdd(&blockTotalNeighbors, neighborIter);
    }

    __syncthreads();

    if(threadIdx.x == 0){
        atomicAdd(&pOutNumNeigbors[0], blockTotalNeighbors);
    }
}

/**
 *  Method to compute the offsets in the neighboring list.
 *  @param  pNumOffsets             Number of offsets.
 *  @param  pNumOffsets             Number of offsets 2.
 *  @param  pOutNeighborsOffsets    List with the offsets of each block.
 *  @param  pOutNeighborsOffsets2   List with the offsets of each block of blocks.
 */
__global__ void computeOffsets(
    const bool pStep1,
    const int pNumOffsets,
    const int pNumOffsets2,
    int* __restrict__ pOutNeighborsOffsets,
    int* __restrict__ pOutNeighborsOffsets2) 
{
    __shared__ int groupOffsets[POINT_BLOCK_PACK_SIZE];

	//Get the local and global counter.
	int currCounter = threadIdx.x;
	int currGlobalCounter = threadIdx.x + blockIdx.x * blockDim.x;

	//Update the shared memory.
	if(currGlobalCounter < pNumOffsets)
		groupOffsets[currCounter] = pOutNeighborsOffsets[currGlobalCounter];
	else
		groupOffsets[currCounter] = 0;

	//SIMD scan.
	for(int i = 1; i <= POINT_BLOCK_PACK_SIZE/2; i*=2)
	{
		__syncthreads();

		//Get the values of the pass.
		int currIndex = currCounter + i;
		int value1 = 0;
		int value2 = 0;
		if(currIndex < POINT_BLOCK_PACK_SIZE){
			value1 = groupOffsets[currCounter];
			value2 = groupOffsets[currIndex];
		}

		__syncthreads();

		//Update with the new value.
		if(currIndex < POINT_BLOCK_PACK_SIZE)
			groupOffsets[currIndex] = value1 + value2;
	}

	__syncthreads();

	//Save the counter into global memory.
	if(currGlobalCounter < pNumOffsets){
		if(currCounter > 0)
			pOutNeighborsOffsets[currGlobalCounter] = groupOffsets[currCounter-1];
		else
			pOutNeighborsOffsets[currGlobalCounter] = 0;
	}

    if(pStep1){
        //Update the offsets buffer.
        if(currCounter == (POINT_BLOCK_PACK_SIZE-1) && blockIdx.x < pNumOffsets2)
            pOutNeighborsOffsets2[blockIdx.x] = groupOffsets[POINT_BLOCK_PACK_SIZE-1];
    }else{
        //Update the second level offset buffer.
        if(currCounter > blockIdx.x && currCounter < pNumOffsets2){
            atomicAdd(&pOutNeighborsOffsets2[currCounter], groupOffsets[POINT_BLOCK_PACK_SIZE-1]);
        }
    }
}

/**
 *  Method to find the neighboring points for each point.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumCells               Number of cells of the grid.
 *  @param  pAABBMinPoint           Minimum point of the grid (3 componenets).
 *  @param  pAABBMaxPoint           Maximum point of the grid (3 componenets).
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pPoints2            List of points from where to find neighbors.
 *  @param  pCellIndexs             Indexs of the grid cells.
 *  @param  pStartIndexsOffset      List with the first level offset to teh start indices.
 *  @param  pStartIndexsOffset2     List with the second level offset to teh start indices.
 *  @param  pStartIndexs            Input/Output parameter with the list of the starting indices in the neighboring list.
 *  @param  pOutNeigbors            Output parameter with the list neighbors of each point.
 */
__global__ void findNeighbors(
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumCells,
    const int pNumNeighbors,
    const float pRadius,
    const float* __restrict__ pAABBMinPoint, 
    const float* __restrict__ pAABBMaxPoint, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const float* __restrict__ pPoints2,
    const int* __restrict__ pCellIndexs,
    const int* __restrict__ pStartIndexsOffset,
    const int* __restrict__ pStartIndexsOffset2,
    int* __restrict__ pStartIndexs,
    int* __restrict__ pOutNeigbors) 
{
    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(currentIndex < pNumPoints){
        int currBatchId = pBatchIds[currentIndex];
        int pointIndex = currentIndex * 3;

        int offsetIndex = currentIndex/POINT_BLOCK_PACK_SIZE;
        int globalOffsetIndex = offsetIndex/POINT_BLOCK_PACK_SIZE;
        int neighborIndex = pStartIndexs[currentIndex]+pStartIndexsOffset[offsetIndex]+pStartIndexsOffset2[globalOffsetIndex];
        pStartIndexs[currentIndex] = neighborIndex;

        float maxAabbSize = max(max(
            pAABBMaxPoint[currBatchId*3] - pAABBMinPoint[currBatchId*3], 
            pAABBMaxPoint[currBatchId*3+1] - pAABBMinPoint[currBatchId*3+1]), 
            pAABBMaxPoint[currBatchId*3+2] - pAABBMinPoint[currBatchId*3+2]);
        float cellSize = maxAabbSize/(float)pNumCells;
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;

        float centralCoords[3] = {pPoints[pointIndex], pPoints[pointIndex+1], pPoints[pointIndex+2]};
        int xCell = max(min((int)floor((centralCoords[0] - pAABBMinPoint[currBatchId*3])/cellSize), pNumCells -1), 0);
        int yCell = max(min((int)floor((centralCoords[1] - pAABBMinPoint[currBatchId*3+1])/cellSize), pNumCells -1), 0);
        int zCell = max(min((int)floor((centralCoords[2] - pAABBMinPoint[currBatchId*3+2])/cellSize), pNumCells -1), 0);

        int neighborIter = 0;
        for(int i = 0; i < 27; ++i)
        {
            int currCellIndex[3] = {xCell+cellOffsets[i][0], yCell+cellOffsets[i][1], zCell+cellOffsets[i][2]};
            if(currCellIndex[0] >= 0 && currCellIndex[0] < pNumCells &&
                currCellIndex[1] >= 0 && currCellIndex[1] < pNumCells &&
                currCellIndex[2] >= 0 && currCellIndex[2] < pNumCells)
            {
                int cellIndexFlat = currBatchId*pNumCells*pNumCells*pNumCells + currCellIndex[0]*pNumCells*pNumCells + currCellIndex[1]*pNumCells + currCellIndex[2];
                int initIndex = pCellIndexs[cellIndexFlat*2];
                int endIndex = pCellIndexs[cellIndexFlat*2 + 1];
                for(int j = initIndex; j < endIndex; ++j)
                {
                    int currPointIndex = j * 3;
                    float currentCoords[3] = {pPoints2[currPointIndex], pPoints2[currPointIndex+1], pPoints2[currPointIndex+2]};
                    float diffVector[3] = {currentCoords[0] - centralCoords[0], currentCoords[1] - centralCoords[1], currentCoords[2] - centralCoords[2]};
                    float pointDist = sqrt(diffVector[0]*diffVector[0] + diffVector[1]*diffVector[1] + diffVector[2]*diffVector[2]);
                    if(pointDist < scaledRadius){
                        pOutNeigbors[neighborIndex*2 + neighborIter] = j;
                        pOutNeigbors[neighborIndex*2 + neighborIter + 1] = currentIndex;
                        neighborIter+=2;
                    }
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

unsigned int countNeighborsCPU(
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumCells,
    const float pRadius,
    const float* pInPts,
    const int* pInBatchIds,
    const float* pInPts2,
    const int* pCellIndexs,
    const float* pAABBMin,
    const float* pAABBMax,
    int* pStartIndex)
{
    //Init device symbols.
    int cellOffsetsCPU[27][3] = {
        {1, 1, 1},{0, 1, 1},{-1, 1, 1},
        {1, 0, 1},{0, 0, 1},{-1, 0, 1},
        {1, -1, 1},{0, -1, 1},{-1, -1, 1},
        {1, 1, 0},{0, 1, 0},{-1, 1, 0},
        {1, 0, 0},{0, 0, 0},{-1, 0, 0},
        {1, -1, 0},{0, -1, 0},{-1, -1, 0},
        {1, 1, -1},{0, 1, -1},{-1, 1, -1},
        {1, 0, -1},{0, 0, -1},{-1, 0, -1},
        {1, -1, -1},{0, -1, -1},{-1, -1, -1}};
    cudaMemcpyToSymbol(cellOffsets, cellOffsetsCPU, 27*3*sizeof(int));

    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;

    //Find the neighbors for each point.
    int* totalNeighbors;
    gpuErrchk(cudaMalloc(&totalNeighbors, sizeof(int)));
    cudaMemset(totalNeighbors, 0, sizeof(int));

    countNeighbors<<<numBlocksPoints, POINT_BLOCK_SIZE>>>(pScaleInv, pNumPoints, pNumCells, 
        pRadius, pAABBMin, pAABBMax, pInPts, pInBatchIds, pInPts2, pCellIndexs, pStartIndex, totalNeighbors);

    gpuErrchk(cudaPeekAtLastError());

    int totalNeighborsCPU = 0;
    cudaMemcpy(&totalNeighborsCPU, totalNeighbors, sizeof(int), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaFree(totalNeighbors));

#ifdef PRINT_CONV_INFO
    printf("Forward Num points: %d | Neighbors: %d\n", pNumPoints, totalNeighborsCPU);
#endif
    
    return totalNeighborsCPU;
        
}  

void computeAuxiliarBuffersSize(
    const int pNumPoints,
    int* PBufferSize1,
    int* PBufferSize2)
{
    (*PBufferSize1) = pNumPoints/POINT_BLOCK_PACK_SIZE;
    (*PBufferSize1) += (pNumPoints%POINT_BLOCK_PACK_SIZE != 0)?1:0;
    (*PBufferSize2) = (*PBufferSize1)/POINT_BLOCK_PACK_SIZE;
    (*PBufferSize2) += ((*PBufferSize1)%POINT_BLOCK_PACK_SIZE != 0)?1:0;
}

void packNeighborsCPU(
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumCells,
    const float pRadius,
    const float* pInPts,
    const int* pInBatchIds,
    const float* pInPts2,
    const int* pCellIndexs,
    const float* pAABBMin,
    const float* pAABBMax,
    int* pAuxBuffOffsets,
    int* pAuxBuffOffsets2,
    int* pStartIndexs,
    int* pPackedIndexs)
{

    //Pack the indexs of the neighbors.
    int numBlocksPointsPack = pNumPoints/POINT_BLOCK_PACK_SIZE;
    numBlocksPointsPack += (pNumPoints%POINT_BLOCK_PACK_SIZE != 0)?1:0;
    int numBlocksPointsPack2 = numBlocksPointsPack/POINT_BLOCK_PACK_SIZE;
    numBlocksPointsPack2 += (numBlocksPointsPack%POINT_BLOCK_PACK_SIZE != 0)?1:0;

    gpuErrchk(cudaMemset(pAuxBuffOffsets, 0, sizeof(int)*numBlocksPointsPack));
    gpuErrchk(cudaMemset(pAuxBuffOffsets2, 0, sizeof(int)*numBlocksPointsPack2));
    
    computeOffsets<<<numBlocksPointsPack, POINT_BLOCK_PACK_SIZE>>>(true, pNumPoints, numBlocksPointsPack, pStartIndexs, pAuxBuffOffsets);
    
    gpuErrchk(cudaPeekAtLastError());

    computeOffsets<<<numBlocksPointsPack2, POINT_BLOCK_PACK_SIZE>>>(false, numBlocksPointsPack, numBlocksPointsPack2, pAuxBuffOffsets, pAuxBuffOffsets2);

    gpuErrchk(cudaPeekAtLastError());

    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;
    findNeighbors<<<numBlocksPoints,POINT_BLOCK_SIZE>>>(pScaleInv, pNumPoints, pNumCells, pNumNeighbors, pRadius, pAABBMin, pAABBMax, 
        pInPts, pInBatchIds, pInPts2, pCellIndexs, pAuxBuffOffsets, pAuxBuffOffsets2, pStartIndexs, pPackedIndexs);

    gpuErrchk(cudaPeekAtLastError());
}
    
