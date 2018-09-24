/////////////////////////////////////////////////////////////////////////////
/// \file compute_pdf.cu
///
/// \brief Cuda implementation of the operation to approximate the  
///        probability distribution function at each sample in the different  
///        receptive fields.
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

#define NEIGHBOR_BLOCK_PDF_SIZE 256

////////////////////////////////////////////////////////////////////////////////// GPU

/**
 *  Method to compute the pdfs of each neighboring point.
 *  @param  pWindow                 Window used to compute the pdfs.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pRadius                 Radius of the convolution.
 *  @param  pAABBMin                Minimum point of the grid (3 componenets).
 *  @param  pAABBMax                Maximum point of the grid (3 componenets).
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pPoints2                List of neighboring points.
 *  @param  pStartIndexs            List of the starting indices in the neighboring list.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pOutPDFs                Output parameter with the pdfs.
 */
__global__ void computePDFs(
    const bool pScaleInv, 
    const float pWindow,
    const int numSamples,
    const int pNumNeighbors,
    const float pRadius,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    float* __restrict__ pOutPDFs) 
{
    int currentNeighborIndex = threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPoint = pNeigbors[neighborIndex];
        float currPointCoords[3] = {pPoints[currentPoint*3], pPoints[currentPoint*3+1], pPoints[currentPoint*3+2]};
        int currBatchId = pBatchIds[currentPoint];

        float maxAabbSize = max(max(
            pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;   
        
        int centralPoint = pNeigbors[neighborIndex+1];
        int initIter = pStartIndexs[centralPoint];
        int endIter = (centralPoint < numSamples-1)?pStartIndexs[centralPoint+1]:pNumNeighbors;

        const float h = pWindow;
        const float invH = 1/h;
        const float invRadH = 1.0/(scaledRadius*h);
        float currPdf = 0.0;
        int iter = initIter;
        while(iter < endIter)
        {
            int iterPoint = pNeigbors[iter*2]*3;
            float iterPointCoords[3] = {pPoints[iterPoint], pPoints[iterPoint+1], pPoints[iterPoint+2]};
            float diff [3] = {
                (iterPointCoords[0] - currPointCoords[0])*invRadH, 
                (iterPointCoords[1] - currPointCoords[1])*invRadH, 
                (iterPointCoords[2] - currPointCoords[2])*invRadH};
            float gaussVal = invH*((0.39894228)*exp((-0.5)*diff[0]*diff[0]));
            gaussVal = gaussVal*invH*((0.39894228)*exp((-0.5)*diff[1]*diff[1]));
            gaussVal = gaussVal*invH*((0.39894228)*exp((-0.5)*diff[2]*diff[2]));
            currPdf += gaussVal;
            iter++;
        }
        
        pOutPDFs[currentNeighborIndex] = (currPdf)/((float)endIter-initIter);
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void computeDPFsCPU(
    const bool scaleInv, 
    const float pWindow, 
    const int numSamples,
    const int pNumNeighbors,
    const float pRadius,
    const float* pInPts,
    const int* pInBatchIds,
    const float* pAABBMin,
    const float* pAABBMax,
    const int* pStartIndexs,
    const int* pPackedIndexs,
    float* pPDFs)
{

    //Compute the PDF.
    dim3 gridDimension = computeBlockGrid(pNumNeighbors, NEIGHBOR_BLOCK_PDF_SIZE);

    computePDFs<<<gridDimension, NEIGHBOR_BLOCK_PDF_SIZE>>>(scaleInv, pWindow, numSamples, pNumNeighbors, 
        pRadius, pAABBMin, pAABBMax, pInPts, pInBatchIds, pStartIndexs, pPackedIndexs, pPDFs);

    gpuErrchk(cudaPeekAtLastError());
}