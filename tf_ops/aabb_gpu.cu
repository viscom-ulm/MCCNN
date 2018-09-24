/////////////////////////////////////////////////////////////////////////////
/// \file aabb_gpu.cu
///
/// \brief Cuda implementation of the operations to compute the axis aligned 
///        bounding box of a batch of point clouds.
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

#define POINT_BLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////// GPU

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 *  Method to compute the bounding box of a point cloud.
 *  @param  pScaleInv       Scale invariance.
 *  @param  pNumPoints      Number of points.
 *  @param  pBatchSize      Size of the batch.
 *  @param  pPoints         List of points.
 *  @param  pBatchIds       List of identifiers of the batch.
 *  @param  pAABBMin        Output parameter with the minimum point of the bounding box.
 *  @param  pAABBMax        Output parameter with the maximum point of the bounding box.
 */
__global__ void comp_AABB(
    const bool pScaleInv,
    const int pNumPoints,
    const int pBatchSize,
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    float* __restrict__ pAABBMin,
    float* __restrict__ pAABBMax) 
{
    extern __shared__ float tmpSharedMemPtr[];

    int currentIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if(currentIndex < pNumPoints){

        if(threadIdx.x < pBatchSize){
            tmpSharedMemPtr[threadIdx.x*3] = FLT_MAX;
            tmpSharedMemPtr[threadIdx.x*3 + 1] = FLT_MAX;
            tmpSharedMemPtr[threadIdx.x*3 + 2] = FLT_MAX;
            tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3] = -FLT_MAX;
            tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3 + 1] = -FLT_MAX;
            tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3 + 2] = -FLT_MAX;
        }

        __syncthreads();

        int batchId = pBatchIds[currentIndex];
        float* aabbMin = &tmpSharedMemPtr[batchId*3];
        float* aabbMax = &tmpSharedMemPtr[pBatchSize*3 + batchId*3];
        
        int pointIndex = currentIndex * 3;
        atomicMin(&aabbMin[0], pPoints[pointIndex]);
        atomicMin(&aabbMin[1], pPoints[pointIndex+1]);
        atomicMin(&aabbMin[2], pPoints[pointIndex+2]);
        atomicMax(&aabbMax[0], pPoints[pointIndex]);
        atomicMax(&aabbMax[1], pPoints[pointIndex+1]);
        atomicMax(&aabbMax[2], pPoints[pointIndex+2]);

        __syncthreads();

        if(threadIdx.x < pBatchSize){
            if(pScaleInv){
                atomicMin(&pAABBMin[threadIdx.x*3], tmpSharedMemPtr[threadIdx.x*3]);
                atomicMin(&pAABBMin[threadIdx.x*3 + 1], tmpSharedMemPtr[threadIdx.x*3 + 1]);
                atomicMin(&pAABBMin[threadIdx.x*3 + 2], tmpSharedMemPtr[threadIdx.x*3 + 2]);
                atomicMax(&pAABBMax[threadIdx.x*3], tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3]);
                atomicMax(&pAABBMax[threadIdx.x*3 + 1], tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3 + 1]);
                atomicMax(&pAABBMax[threadIdx.x*3 + 2], tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3 + 2]);
            }else{
                for(int i = 0; i < pBatchSize; ++i)
                {
                    atomicMin(&pAABBMin[i*3], tmpSharedMemPtr[threadIdx.x*3]);
                    atomicMin(&pAABBMin[i*3 + 1], tmpSharedMemPtr[threadIdx.x*3 + 1]);
                    atomicMin(&pAABBMin[i*3 + 2], tmpSharedMemPtr[threadIdx.x*3 + 2]);
                    atomicMax(&pAABBMax[i*3], tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3]);
                    atomicMax(&pAABBMax[i*3 + 1], tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3 + 1]);
                    atomicMax(&pAABBMax[i*3 + 2], tmpSharedMemPtr[pBatchSize*3 + threadIdx.x*3 + 2]);
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void computeAABB(
    const bool pScaleInv,
    const int pNumPoints,
    const int pBatchSize,
    const float* pPoints,
    const int* pBatchIds,
    float* pAABBMin,
    float* pAABBMax)
{
    float maxFlt[pBatchSize*3];
    float minFlt[pBatchSize*3];
    for(int i = 0; i < pBatchSize*3; ++i){
        maxFlt[i] = FLT_MAX;
        minFlt[i] = -FLT_MAX;
    }
    gpuErrchk(cudaMemcpy(pAABBMin, &maxFlt[0], pBatchSize*3*sizeof(float),  cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(pAABBMax, &minFlt[0], pBatchSize*3*sizeof(float),  cudaMemcpyHostToDevice));
    int numBlocksPoints = pNumPoints/POINT_BLOCK_SIZE;
    numBlocksPoints += (pNumPoints%POINT_BLOCK_SIZE != 0)?1:0;
    comp_AABB<<<numBlocksPoints, POINT_BLOCK_SIZE, pBatchSize*6*sizeof(float)>>>(pScaleInv, pNumPoints, pBatchSize, pPoints, pBatchIds, pAABBMin, pAABBMax);
}