/////////////////////////////////////////////////////////////////////////////
/// \file spatial_conv.cu
///
/// \brief Cuda implementation of the operations to perform a spatial 
///        convolution on a batch of point clouds.
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

#define EXECUTION_BLOCK_MLP_SIZE 128

////////////////////////////////////////////////////////////////////////////////// GPU

__device__ void evaluateMLP(
    const int pThreadId,
    const int pOffset,
    const int pTotalBlocks,
    const int pNumFeatures,
    const int pNumOutFeatures,
    const int pNumNeuronsOut,
    const int pFeatureIndex,
    const int pOutFeatureIndex,
    const float pNumSamples,
    const float pCurrentPDF,
    const float pCurrPointCoords[3],
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pFeatures,
    float* __restrict__ pTmpVector1,
    float* __restrict__ pTmpVector2,
    float* __restrict__ pOutFeatures)
{
    //Compute output first layer.

    pTmpVector1[pThreadId] = max(pCurrPointCoords[0]*pWeightsHidd1[pThreadId*3] + 
                        pCurrPointCoords[1]*pWeightsHidd1[pThreadId*3 + 1] +
                        pCurrPointCoords[2]*pWeightsHidd1[pThreadId*3 + 2] +
                        pBiasHidd1[pThreadId], 0.0);

    __syncthreads();

    //Compute output second layer.
    float auxResult = 0.0;
    for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
    {
        auxResult += pTmpVector1[j]*pWeightsHidd2[pThreadId*BLOCK_MLP_SIZE + j];
    }
    pTmpVector2[pThreadId] = max(auxResult + pBiasHidd2[pThreadId], 0.0);

    __syncthreads();
    
    //Compute output layer.
    if((pOffset+pThreadId) < pNumNeuronsOut){
        auxResult = 0.0;
        for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
        {
            auxResult += pTmpVector2[j]*pWeightsOut[pThreadId*BLOCK_MLP_SIZE + j];  
        }
        auxResult = auxResult + pBiasOut[pThreadId];
        int currInFeatureIndex = (pOffset+pThreadId)%pNumFeatures;
        int currOutFeatureIndex = (pOffset+pThreadId)/pNumFeatures;
        atomicAdd(&pOutFeatures[pOutFeatureIndex+currOutFeatureIndex], 
            (pFeatures[pFeatureIndex+currInFeatureIndex]*auxResult)/(pCurrentPDF*pNumSamples));
    }
}

/**
 *  Method to evaluate the MLP.
 *  @param  pAVG                    Boolean that indicates if the results is divided by the number of neighbors or not.
 *  @param  pScaleInv               Boolean that indicates if the radius is defined relative to the bounding box.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pNumFeatures            Number of input features per point.
 *  @param  pNumOutFeatures         Number of output features per point.
 *  @param  pRadius                 Radius of the convolution.
 *  @param  pWeightsHidd1           Weights of the neurons in the first hidden layer.
 *  @param  pWeightsHidd2           Weights of the neurons in the second hidden layer.
 *  @param  pWeightsOut             Weights of the neurons in the output layer.
 *  @param  pBiasHidd1              Biases of the neurons in the first hidden layer.
 *  @param  pBiasHidd2              Biases of the neurons in the second hidden layer.
 *  @param  pBiasOut                Biases of the neurons in the output layer.
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pFeatures               List of input features.
 *  @param  pStartIndexs            List of start indices for each point.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pPDFs                   List of the pdf values.
 *  @param  pOutFeatures            Output parameter with the list of output features.
 */
__global__ void evaluateMLPKernel(
    const bool pAvg,
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumFeatures,
    const int pNumOutFeatures,
    const float pRadius,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pSamples, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const float* __restrict__ pFeatures,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pPDFs,
    float* __restrict__ pOutFeatures) 
{
    extern __shared__ float mlpIntermediateRes[];

    int neuronsOut = pNumOutFeatures*pNumFeatures;
    int numBlocksXNeigh = neuronsOut/BLOCK_MLP_SIZE;
    numBlocksXNeigh += (neuronsOut%BLOCK_MLP_SIZE != 0)?1:0;

    unsigned long long int currentIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    int currentNeighborIndex = currentIndex/(numBlocksXNeigh*BLOCK_MLP_SIZE);
    int offset = currentIndex%(numBlocksXNeigh*BLOCK_MLP_SIZE);
    offset = offset - offset%BLOCK_MLP_SIZE;
    int threadId = threadIdx.x%BLOCK_MLP_SIZE;
    int threadOffset = threadIdx.x - threadId;

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];
        
        int currBatchId = pBatchIds[currentPointIndex];
        float maxAabbSize = max(max(
            pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;
        
        float currPointCoords[3] = {
            (pPoints[currentPointIndex*3] - pSamples[centralPointIndex*3])/scaledRadius, 
            (pPoints[currentPointIndex*3+1] - pSamples[centralPointIndex*3+1])/scaledRadius, 
            (pPoints[currentPointIndex*3+2] - pSamples[centralPointIndex*3+2])/scaledRadius};
        float currPDF = pPDFs[currentNeighborIndex];
        int initIter = pStartIndexs[centralPointIndex];
        int endIter = (centralPointIndex < pNumPoints-1)?pStartIndexs[centralPointIndex+1]:pNumNeighbors;
        float numNeighbors = (pAvg)?(float)(endIter-initIter):1.0;
        int featureIndex = currentPointIndex*pNumFeatures;
        int outFeatureIndex = centralPointIndex*pNumOutFeatures;

        float* temporalMemory1 = &mlpIntermediateRes[threadOffset];
        float* temporalMemory2 = &mlpIntermediateRes[EXECUTION_BLOCK_MLP_SIZE + threadOffset];

        evaluateMLP(threadId, offset, numBlocksXNeigh, pNumFeatures, pNumOutFeatures, neuronsOut, 
            featureIndex, outFeatureIndex, numNeighbors, currPDF, currPointCoords, 
            &pWeightsHidd1[offset*3], &pWeightsHidd2[offset*BLOCK_MLP_SIZE], &pWeightsOut[offset*BLOCK_MLP_SIZE], 
            &pBiasHidd1[offset], &pBiasHidd2[offset], &pBiasOut[offset], 
            pFeatures, temporalMemory1, temporalMemory2, pOutFeatures);
    }
}

__device__ void evaluateMLPNoComb(
    const int pThreadId,
    const int pOffset,
    const int pTotalBlocks,
    const int pNumFeatures,
    const int pFeatureIndex,
    const int pOutFeatureIndex,
    const float pNumSamples,
    const float pCurrentPDF,
    const float pCurrPointCoords[3],
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pFeatures,
    float* __restrict__ pTmpVector1,
    float* __restrict__ pTmpVector2,
    float* __restrict__ pOutFeatures)
{
    //Compute output first layer.

    pTmpVector1[pThreadId] = max(pCurrPointCoords[0]*pWeightsHidd1[pThreadId*3] + 
                        pCurrPointCoords[1]*pWeightsHidd1[pThreadId*3 + 1] +
                        pCurrPointCoords[2]*pWeightsHidd1[pThreadId*3 + 2] +
                        pBiasHidd1[pThreadId], 0.0);

    __syncthreads();

    //Compute output second layer.
    float auxResult = 0.0;
    for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
    {
        auxResult += pTmpVector1[j]*pWeightsHidd2[pThreadId*BLOCK_MLP_SIZE + j];
    }
    pTmpVector2[pThreadId] = max(auxResult + pBiasHidd2[pThreadId], 0.0);

    __syncthreads();
    
    //Compute output layer.
    if((pOffset+pThreadId) < pNumFeatures){
        auxResult = 0.0;
        for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
        {
            auxResult += pTmpVector2[j]*pWeightsOut[pThreadId*BLOCK_MLP_SIZE + j];  
        }
        auxResult = auxResult + pBiasOut[pThreadId];
        int currFeatureIndex = (pOffset+pThreadId)%pNumFeatures;
        atomicAdd(&pOutFeatures[pOutFeatureIndex+currFeatureIndex], 
            (pFeatures[pFeatureIndex+currFeatureIndex]*auxResult)/(pCurrentPDF*pNumSamples));
    }
}

/**
 *  Method to evaluate the MLP.
 *  @param  pAVG                    Boolean that indicates if the results is divided by the number of neighbors or not.
 *  @param  pScaleInv               Boolean that indicates if the radius is defined relative to the bounding box.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pNumFeatures            Number of input features per point.
 *  @param  pRadius                 Radius of the convolution.
 *  @param  pWeightsHidd1           Weights of the neurons in the first hidden layer.
 *  @param  pWeightsHidd2           Weights of the neurons in the second hidden layer.
 *  @param  pWeightsOut             Weights of the neurons in the output layer.
 *  @param  pBiasHidd1              Biases of the neurons in the first hidden layer.
 *  @param  pBiasHidd2              Biases of the neurons in the second hidden layer.
 *  @param  pBiasOut                Biases of the neurons in the output layer.
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pFeatures               List of input features.
 *  @param  pStartIndexs            List of start indices for each point.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pPDFs                   List of the pdf values.
 *  @param  pOutFeatures            Output parameter with the list of output features.
 */
__global__ void evaluateMLPNoCombinKernel(
    const bool pAvg,
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumFeatures,
    const float pRadius,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pSamples, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const float* __restrict__ pFeatures,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pPDFs,
    float* __restrict__ pOutFeatures) 
{
    extern __shared__ float mlpIntermediateRes[];

    int neuronsOut = pNumFeatures;
    int numBlocksXNeigh = neuronsOut/BLOCK_MLP_SIZE;
    numBlocksXNeigh += (neuronsOut%BLOCK_MLP_SIZE != 0)?1:0;

    unsigned long long int currentIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    int currentNeighborIndex = currentIndex/(numBlocksXNeigh*BLOCK_MLP_SIZE);
    int offset = currentIndex%(numBlocksXNeigh*BLOCK_MLP_SIZE);
    offset = offset - offset%BLOCK_MLP_SIZE;
    int threadId = threadIdx.x%BLOCK_MLP_SIZE;
    int threadOffset = threadIdx.x - threadId;

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];

        int currBatchId = pBatchIds[currentPointIndex];
        float maxAabbSize = max(max(
            pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;
        
        float currPointCoords[3] = {
            (pPoints[currentPointIndex*3] - pSamples[centralPointIndex*3])/scaledRadius, 
            (pPoints[currentPointIndex*3+1] - pSamples[centralPointIndex*3+1])/scaledRadius, 
            (pPoints[currentPointIndex*3+2] - pSamples[centralPointIndex*3+2])/scaledRadius};
        float currPDF = pPDFs[currentNeighborIndex];
        int initIter = pStartIndexs[centralPointIndex];
        int endIter = (centralPointIndex < pNumPoints-1)?pStartIndexs[centralPointIndex+1]:pNumNeighbors;
        float numNeighbors = (pAvg)?(float)(endIter-initIter):1.0;
        int featureIndex = currentPointIndex*pNumFeatures;
        int outFeatureIndex = centralPointIndex*pNumFeatures;
        
        float* temporalMemory1 = &mlpIntermediateRes[threadOffset];
        float* temporalMemory2 = &mlpIntermediateRes[EXECUTION_BLOCK_MLP_SIZE + threadOffset];

        evaluateMLPNoComb(threadId, offset, numBlocksXNeigh, pNumFeatures,
            featureIndex, outFeatureIndex, numNeighbors, currPDF, currPointCoords, 
            &pWeightsHidd1[offset*3], &pWeightsHidd2[offset*BLOCK_MLP_SIZE], &pWeightsOut[offset*BLOCK_MLP_SIZE], 
            &pBiasHidd1[offset], &pBiasHidd2[offset], &pBiasOut[offset], 
            pFeatures, temporalMemory1, temporalMemory2, pOutFeatures);
    }
}

__device__ void computedconvj_d(
    const int pThreadId,
    const int pOffset,
    const int pTotalBlocks,
    const int pNumNeuronsOut,
    const int pNumFeatures,
    const int pPointIndex,
    const int pCentralPointIndex,
    const int pInFeatureIndex,
    const int pOutFeatureIndex,
    const float pNumSamples,
    const float pCurrentPDF,
    const float pCurrPointCoords[3],
    const float pRadius,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pInFeatures,
    const float* __restrict__ pOutFeatureGradients,
    float* __restrict__ pTmpVector1,
    float* __restrict__ pTmpVector2,
    float* __restrict__ pTmpVector3,
    float* __restrict__ pTmpVector4,
    float* __restrict__ pWeightsHidd1Grad,
    float* __restrict__ pWeightsHidd2Grad,
    float* __restrict__ pWeightsOutGrad,
    float* __restrict__ pBiasHidd1Grad,
    float* __restrict__ pBiasHidd2Grad,
    float* __restrict__ pBiasOutGrad,
    float* __restrict__ pFeaturesGrads)
{    
    //Compute output first layer.

    pTmpVector1[pThreadId] = pCurrPointCoords[0]*pWeightsHidd1[pThreadId*3] + 
                        pCurrPointCoords[1]*pWeightsHidd1[pThreadId*3 + 1] +
                        pCurrPointCoords[2]*pWeightsHidd1[pThreadId*3 + 2] +
                        pBiasHidd1[pThreadId];

    __syncthreads();

    //Compute output second layer.
    float auxResult = 0.0;
    for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
    {
        auxResult += max(pTmpVector1[j], 0.0)*pWeightsHidd2[pThreadId*BLOCK_MLP_SIZE + j];
    }
    pTmpVector2[pThreadId] = auxResult +pBiasHidd2[pThreadId];

    __syncthreads();

    //Gradients computation    
    
    //Gradients of the output layer parameters w and b and in features.
    if((pOffset+pThreadId) < pNumNeuronsOut){
        int currInFeatureIndex = (pOffset+pThreadId)%pNumFeatures;
        int currOutFeatureIndex = (pOffset+pThreadId)/pNumFeatures;
        float currFeature = pInFeatures[pInFeatureIndex + currInFeatureIndex];
        float outGradient = pOutFeatureGradients[pOutFeatureIndex+currOutFeatureIndex];
        float commonFactor = (currFeature*outGradient)/(pCurrentPDF*pNumSamples);
        auxResult = 0.0;
        for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
        {
            atomicAdd(&pWeightsOutGrad[pThreadId*BLOCK_MLP_SIZE + j], commonFactor*max(pTmpVector2[j], 0.0));
            auxResult += max(pTmpVector2[j], 0.0)*pWeightsOut[pThreadId*BLOCK_MLP_SIZE + j];
        }
        atomicAdd(&pBiasOutGrad[pThreadId], commonFactor);

        //In features gradient update.
        auxResult = auxResult + pBiasOut[pThreadId];
        atomicAdd(&pFeaturesGrads[pInFeatureIndex + currInFeatureIndex], outGradient*auxResult/(pCurrentPDF*pNumSamples));
    }

    //Gradients of the second hiddend layer.
    auxResult = 0.0;
    float commonFactor = (pTmpVector2[pThreadId] >= 0.0)?1.0:0.0;
    int numOutsBlock = min(pNumNeuronsOut - pOffset, BLOCK_MLP_SIZE);
    for(int i = 0; i < numOutsBlock; ++i)
    {
        int currInFeatureIndex = (i+pOffset)%pNumFeatures;
        int currOutFeatureIndex = (i+pOffset)/pNumFeatures;
        float currFeature = pInFeatures[pInFeatureIndex + currInFeatureIndex];
        float outGradient = pOutFeatureGradients[pOutFeatureIndex+currOutFeatureIndex];
        auxResult += outGradient*currFeature*pWeightsOut[pThreadId + i*BLOCK_MLP_SIZE];
    }
    pTmpVector3[pThreadId] = (commonFactor*auxResult)/(pCurrentPDF*pNumSamples);

    __syncthreads();

    //Gradients of the second hiddend layer parameters w and b.
    commonFactor = pTmpVector3[pThreadId];
    for(int i = 0; i < BLOCK_MLP_SIZE; ++i)
    {
        atomicAdd(&pWeightsHidd2Grad[pThreadId*BLOCK_MLP_SIZE + i], 
                commonFactor*max(pTmpVector1[i], 0.0)); 
    }
    atomicAdd(&pBiasHidd2Grad[pThreadId], commonFactor);

    //Gradients of the first hiddend layer.
    auxResult = 0.0;
    commonFactor = (pTmpVector1[pThreadId] >= 0.0)?1.0:0.0;
    for(int i = 0; i < BLOCK_MLP_SIZE; ++i)
    {
        auxResult += pTmpVector3[i]*pWeightsHidd2[pThreadId+ i*BLOCK_MLP_SIZE];
    }
    pTmpVector4[pThreadId] = commonFactor*auxResult;

    __syncthreads();

    //Gradients of the first hiddend layer parameters w and b.
    commonFactor = pTmpVector4[pThreadId];
    for(int i = 0; i < 3; ++i)
    {
        atomicAdd(&pWeightsHidd1Grad[pThreadId*3 + i], commonFactor*pCurrPointCoords[i]); 
    }
    atomicAdd(&pBiasHidd1Grad[pThreadId], commonFactor);
}

/**
 *  Method to evaluate the MLP.
 *  @param  pAVG                    Boolean that indicates if the results is divided by the number of neighbors or not.
 *  @param  pScaleInv               Boolean that indicates if the radius is defined relative to the bounding box.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pNumFeatures            Number of input features per point.
 *  @param  pNumOutFeatures         Number of output features per point.
 *  @param  pRadius                 Radius of the convolution.
 *  @param  pWeightsHidd1           Weights of the neurons in the first hidden layer.
 *  @param  pWeightsHidd2           Weights of the neurons in the second hidden layer.
 *  @param  pWeightsOut             Weights of the neurons in the output layer.
 *  @param  pBiasHidd1              Biases of the neurons in the first hidden layer.
 *  @param  pBiasHidd2              Biases of the neurons in the second hidden layer.
 *  @param  pBiasOut                Biases of the neurons in the output layer.
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pFeatures               List of input features.
 *  @param  pOutFeaturesGrads       Gradients of the output convolutions.
 *  @param  pStartIndexs            List of start indices for each point.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pPDFs                   List of the pdf values.
 *  @param  pWeightsHidd1Grads      Output parameter with the list of gradients for the weights of the first hidden layer.
 *  @param  pWeightsHidd2Grads      Output parameter with the list of gradients for the weights of the second hidden layer.
 *  @param  pWeightsOutGrads        Output parameter with the list of gradients for the weights of the outpu layer.
 *  @param  pBiasHidd1Grads         Output parameter with the list of gradients for the biases of the first hidden layer.
 *  @param  pBiasHidd2Grads         Output parameter with the list of gradients for the biases of the second hidden layer.
 *  @param  pBiasOutGrads           Output parameter with the list of gradients for the biases of the output layer.
 *  @param  pPointsGrads            Output parameter with the list of gradients for the points.
 */
__global__ void computedconvj_dKernel(
    const bool pAvg,
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumFeatures,
    const int pNumOutFeatures,
    const float pRadius,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pSamples, 
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const float* __restrict__ pFeatures,
    const float* __restrict__ pOutFeaturesGrads,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pPDFs,
    float* __restrict__ pWeightsHidd1Grads,
    float* __restrict__ pWeightsHidd2Grads,
    float* __restrict__ pWeightsOutGrads,
    float* __restrict__ pBiasHidd1Grads,
    float* __restrict__ pBiasHidd2Grads,
    float* __restrict__ pBiasOutGrads,
    float* __restrict__ pFeaturesGrads) 
{
    extern __shared__ float mlpdconvjIntermediateRes[];

    int neuronsOut = pNumOutFeatures*pNumFeatures;
    int numBlocksXNeigh = neuronsOut/BLOCK_MLP_SIZE;
    numBlocksXNeigh += (neuronsOut%BLOCK_MLP_SIZE != 0)?1:0;

    unsigned long long int currentIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    int currentNeighborIndex = currentIndex/(numBlocksXNeigh*BLOCK_MLP_SIZE);
    int offset = currentIndex%(numBlocksXNeigh*BLOCK_MLP_SIZE);
    offset = offset - offset%BLOCK_MLP_SIZE;
    int threadId = threadIdx.x%BLOCK_MLP_SIZE;
    int threadOffset = threadIdx.x - threadId;

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];
        int currBatchId = pBatchIds[currentPointIndex];

        float maxAabbSize = max(max(
            pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
            
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;
        float currPointCoords[3] = {
            (pPoints[currentPointIndex*3] - pSamples[centralPointIndex*3])/scaledRadius, 
            (pPoints[currentPointIndex*3+1] - pSamples[centralPointIndex*3+1])/scaledRadius, 
            (pPoints[currentPointIndex*3+2] - pSamples[centralPointIndex*3+2])/scaledRadius};
        float currPDF = pPDFs[currentNeighborIndex];
        int initIter = pStartIndexs[centralPointIndex];
        int endIter = (centralPointIndex < pNumPoints-1)?pStartIndexs[centralPointIndex+1]:pNumNeighbors;
        float numNeighbors = (pAvg)?(float)(endIter-initIter):1.0;
        int featureIndex = currentPointIndex*pNumFeatures;
        int outFeatureIndex = centralPointIndex*pNumOutFeatures;

        float* temporalMemory1 = &(mlpdconvjIntermediateRes[threadOffset]);
        float* temporalMemory2 = &(mlpdconvjIntermediateRes[EXECUTION_BLOCK_MLP_SIZE + threadOffset]);
        float* temporalMemory3 = &(mlpdconvjIntermediateRes[EXECUTION_BLOCK_MLP_SIZE*2 + threadOffset]);
        float* temporalMemory4 = &(mlpdconvjIntermediateRes[EXECUTION_BLOCK_MLP_SIZE*3 + threadOffset]);

        computedconvj_d(threadId, offset, numBlocksXNeigh, neuronsOut, pNumFeatures, currentPointIndex, centralPointIndex,
            featureIndex, outFeatureIndex, numNeighbors, currPDF, currPointCoords, scaledRadius, 
            &pWeightsHidd1[offset*3], &pWeightsHidd2[offset*BLOCK_MLP_SIZE], &pWeightsOut[offset*BLOCK_MLP_SIZE], 
            &pBiasHidd1[offset], &pBiasHidd2[offset], &pBiasOut[offset],
            pFeatures, pOutFeaturesGrads, temporalMemory1, temporalMemory2, temporalMemory3, temporalMemory4, 
            &pWeightsHidd1Grads[offset*3], &pWeightsHidd2Grads[offset*BLOCK_MLP_SIZE], 
            &pWeightsOutGrads[offset*BLOCK_MLP_SIZE], &pBiasHidd1Grads[offset],
            &pBiasHidd2Grads[offset], &pBiasOutGrads[offset], pFeaturesGrads);
    }
}

__device__ void computedconvj_dNoCombin(
    const int pThreadId,
    const int pOffset,
    const int pTotalBlocks,
    const int pNumFeatures,
    const int pPointIndex,
    const int pCentralPointIndex,
    const int pInFeatureIndex,
    const int pOutFeatureIndex,
    const float pNumSamples,
    const float pCurrentPDF,
    const float pCurrPointCoords[3],
    const float pRadius,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pInFeatures,
    const float* __restrict__ pOutFeatureGradients,
    float* __restrict__ pTmpVector1,
    float* __restrict__ pTmpVector2,
    float* __restrict__ pTmpVector3,
    float* __restrict__ pTmpVector4,
    float* __restrict__ pWeightsHidd1Grad,
    float* __restrict__ pWeightsHidd2Grad,
    float* __restrict__ pWeightsOutGrad,
    float* __restrict__ pBiasHidd1Grad,
    float* __restrict__ pBiasHidd2Grad,
    float* __restrict__ pBiasOutGrad,
    float* __restrict__ pFeaturesGrads)
{    
    //Compute output first layer.

    pTmpVector1[pThreadId] = pCurrPointCoords[0]*pWeightsHidd1[pThreadId*3] + 
                        pCurrPointCoords[1]*pWeightsHidd1[pThreadId*3 + 1] +
                        pCurrPointCoords[2]*pWeightsHidd1[pThreadId*3 + 2] +
                        pBiasHidd1[pThreadId];

    __syncthreads();

    //Compute output second layer.
    float auxResult = 0.0;
    for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
    {
        auxResult += max(pTmpVector1[j], 0.0)*pWeightsHidd2[pThreadId*BLOCK_MLP_SIZE + j];
    }
    pTmpVector2[pThreadId] = auxResult +pBiasHidd2[pThreadId];

    __syncthreads();

    //Gradients computation    
    
    //Gradients of the output layer parameters w and b and in features.
    if((pOffset+pThreadId) < pNumFeatures){
        int currFeatureIndex = (pOffset+pThreadId)%pNumFeatures;
        float currFeature = pInFeatures[pInFeatureIndex + currFeatureIndex];
        float outGradient = pOutFeatureGradients[pOutFeatureIndex+currFeatureIndex];
        float commonFactor = (currFeature*outGradient)/(pCurrentPDF*pNumSamples);
        auxResult = 0.0;
        for(int j = 0; j < BLOCK_MLP_SIZE; ++j)
        {
            atomicAdd(&pWeightsOutGrad[pThreadId*BLOCK_MLP_SIZE + j], commonFactor*max(pTmpVector2[j], 0.0));
            auxResult += max(pTmpVector2[j], 0.0)*pWeightsOut[pThreadId*BLOCK_MLP_SIZE + j];
        }
        atomicAdd(&pBiasOutGrad[pThreadId], commonFactor);

        //In features gradient update.
        auxResult = auxResult + pBiasOut[pThreadId];
        atomicAdd(&pFeaturesGrads[pInFeatureIndex + currFeatureIndex], outGradient*auxResult/(pCurrentPDF*pNumSamples));
    }

    //Gradients of the second hiddend layer.
    auxResult = 0.0;
    float commonFactor = (pTmpVector2[pThreadId] >= 0.0)?1.0:0.0;
    int numOutsBlock = min(pNumFeatures - pOffset, BLOCK_MLP_SIZE);
    for(int i = 0; i < numOutsBlock; ++i)
    {
        int currFeatureIndex = (i+pOffset)%pNumFeatures;
        float currFeature = pInFeatures[pInFeatureIndex + currFeatureIndex];
        float outGradient = pOutFeatureGradients[pOutFeatureIndex+currFeatureIndex];
        auxResult += outGradient*currFeature*pWeightsOut[pThreadId + i*BLOCK_MLP_SIZE];
    }
    pTmpVector3[pThreadId] = (commonFactor*auxResult)/(pCurrentPDF*pNumSamples);

    __syncthreads();

    //Gradients of the second hiddend layer parameters w and b.
    commonFactor = pTmpVector3[pThreadId];
    for(int i = 0; i < BLOCK_MLP_SIZE; ++i)
    {
        atomicAdd(&pWeightsHidd2Grad[pThreadId*BLOCK_MLP_SIZE + i], 
                commonFactor*max(pTmpVector1[i], 0.0)); 
    }
    atomicAdd(&pBiasHidd2Grad[pThreadId], commonFactor);

    //Gradients of the first hiddend layer.
    auxResult = 0.0;
    commonFactor = (pTmpVector1[pThreadId] >= 0.0)?1.0:0.0;
    for(int i = 0; i < BLOCK_MLP_SIZE; ++i)
    {
        auxResult += pTmpVector3[i]*pWeightsHidd2[pThreadId+ i*BLOCK_MLP_SIZE];
    }
    pTmpVector4[pThreadId] = commonFactor*auxResult;

    __syncthreads();

    //Gradients of the first hiddend layer parameters w and b.
    commonFactor = pTmpVector4[pThreadId];
    for(int i = 0; i < 3; ++i)
    {
        atomicAdd(&pWeightsHidd1Grad[pThreadId*3 + i], commonFactor*pCurrPointCoords[i]); 
    }
    atomicAdd(&pBiasHidd1Grad[pThreadId], commonFactor);
}

/**
 *  Method to evaluate the MLP.
 *  @param  pAVG                    Boolean that indicates if the results is divided by the number of neighbors or not.
 *  @param  pScaleInv               Boolean that indicates if the radius is defined relative to the bounding box.
 *  @param  pNumPoints              Number of points.
 *  @param  pNumNeighbors           Number of neighboring points.
 *  @param  pNumFeatures            Number of input features per point.
 *  @param  pRadius                 Radius of the convolution.
 *  @param  pWeightsHidd1           Weights of the neurons in the first hidden layer.
 *  @param  pWeightsHidd2           Weights of the neurons in the second hidden layer.
 *  @param  pWeightsOut             Weights of the neurons in the output layer.
 *  @param  pBiasHidd1              Biases of the neurons in the first hidden layer.
 *  @param  pBiasHidd2              Biases of the neurons in the second hidden layer.
 *  @param  pBiasOut                Biases of the neurons in the output layer.
 *  @param  pPoints                 List of points.
 *  @param  pBatchIds               List of batch ids.
 *  @param  pFeatures               List of input features.
 *  @param  pOutFeaturesGrads       Gradients of the output convolutions.
 *  @param  pStartIndexs            List of start indices for each point.
 *  @param  pNeigbors               List neighbors of each point.
 *  @param  pPDFs                   List of the pdf values.
 *  @param  pWeightsHidd1Grads      Output parameter with the list of gradients for the weights of the first hidden layer.
 *  @param  pWeightsHidd2Grads      Output parameter with the list of gradients for the weights of the second hidden layer.
 *  @param  pWeightsOutGrads        Output parameter with the list of gradients for the weights of the outpu layer.
 *  @param  pBiasHidd1Grads         Output parameter with the list of gradients for the biases of the first hidden layer.
 *  @param  pBiasHidd2Grads         Output parameter with the list of gradients for the biases of the second hidden layer.
 *  @param  pBiasOutGrads           Output parameter with the list of gradients for the biases of the output layer.
 *  @param  pPointsGrads            Output parameter with the list of gradients for the points.
 */
__global__ void computedconvj_dNoCombinKernel(
    const bool pAvg,
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumFeatures,
    const float pRadius,
    const float* __restrict__ pAABBMin,
    const float* __restrict__ pAABBMax,
    const float* __restrict__ pWeightsHidd1,
    const float* __restrict__ pWeightsHidd2,
    const float* __restrict__ pWeightsOut,
    const float* __restrict__ pBiasHidd1,
    const float* __restrict__ pBiasHidd2,
    const float* __restrict__ pBiasOut,
    const float* __restrict__ pSamples,
    const float* __restrict__ pPoints, 
    const int* __restrict__ pBatchIds, 
    const float* __restrict__ pFeatures,
    const float* __restrict__ pOutFeaturesGrads,
    const int* __restrict__ pStartIndexs,
    const int* __restrict__ pNeigbors,
    const float* __restrict__ pPDFs,
    float* __restrict__ pWeightsHidd1Grads,
    float* __restrict__ pWeightsHidd2Grads,
    float* __restrict__ pWeightsOutGrads,
    float* __restrict__ pBiasHidd1Grads,
    float* __restrict__ pBiasHidd2Grads,
    float* __restrict__ pBiasOutGrads,
    float* __restrict__ pFeaturesGrads) 
{
    extern __shared__ float mlpdconvjIntermediateRes[];

    int neuronsOut = pNumFeatures;
    int numBlocksXNeigh = neuronsOut/BLOCK_MLP_SIZE;
    numBlocksXNeigh += (neuronsOut%BLOCK_MLP_SIZE != 0)?1:0;

    unsigned long long int currentIndex = threadIdx.x + 
        blockDim.x*(blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.x*gridDim.y);
    int currentNeighborIndex = currentIndex/(numBlocksXNeigh*BLOCK_MLP_SIZE);
    int offset = currentIndex%(numBlocksXNeigh*BLOCK_MLP_SIZE);
    offset = offset - offset%BLOCK_MLP_SIZE;
    int threadId = threadIdx.x%BLOCK_MLP_SIZE;
    int threadOffset = threadIdx.x - threadId;

    if(currentNeighborIndex < pNumNeighbors){

        int neighborIndex = currentNeighborIndex * 2;
        int currentPointIndex = pNeigbors[neighborIndex];
        int centralPointIndex = pNeigbors[neighborIndex+1];
        int currBatchId = pBatchIds[currentPointIndex];

        float maxAabbSize = max(max(
            pAABBMax[currBatchId*3] - pAABBMin[currBatchId*3], 
            pAABBMax[currBatchId*3+1] - pAABBMin[currBatchId*3+1]), 
            pAABBMax[currBatchId*3+2] - pAABBMin[currBatchId*3+2]);
        float scaledRadius = (pScaleInv)?pRadius*maxAabbSize:pRadius;

        float currPointCoords[3] = {
            (pPoints[currentPointIndex*3] - pSamples[centralPointIndex*3])/scaledRadius, 
            (pPoints[currentPointIndex*3+1] - pSamples[centralPointIndex*3+1])/scaledRadius, 
            (pPoints[currentPointIndex*3+2] - pSamples[centralPointIndex*3+2])/scaledRadius};
        float currPDF = pPDFs[currentNeighborIndex];
        int initIter = pStartIndexs[centralPointIndex];
        int endIter = (centralPointIndex < pNumPoints-1)?pStartIndexs[centralPointIndex+1]:pNumNeighbors;
        float numNeighbors = (pAvg)?(float)(endIter-initIter):1.0;
        int featureIndex = currentPointIndex*pNumFeatures;
        int outFeatureIndex = centralPointIndex*pNumFeatures;

        float* temporalMemory1 = &(mlpdconvjIntermediateRes[threadOffset]);
        float* temporalMemory2 = &(mlpdconvjIntermediateRes[EXECUTION_BLOCK_MLP_SIZE + threadOffset]);
        float* temporalMemory3 = &(mlpdconvjIntermediateRes[EXECUTION_BLOCK_MLP_SIZE*2 + threadOffset]);
        float* temporalMemory4 = &(mlpdconvjIntermediateRes[EXECUTION_BLOCK_MLP_SIZE*3 + threadOffset]);

        computedconvj_dNoCombin(threadId, offset, numBlocksXNeigh, pNumFeatures, currentPointIndex, centralPointIndex,
            featureIndex, outFeatureIndex, numNeighbors, currPDF, currPointCoords, scaledRadius, 
            &pWeightsHidd1[offset*3], &pWeightsHidd2[offset*BLOCK_MLP_SIZE], &pWeightsOut[offset*BLOCK_MLP_SIZE], 
            &pBiasHidd1[offset], &pBiasHidd2[offset], &pBiasOut[offset],
            pFeatures, pOutFeaturesGrads, temporalMemory1, temporalMemory2, temporalMemory3, temporalMemory4, 
            &pWeightsHidd1Grads[offset*3], &pWeightsHidd2Grads[offset*BLOCK_MLP_SIZE], 
            &pWeightsOutGrads[offset*BLOCK_MLP_SIZE], &pBiasHidd1Grads[offset],
            &pBiasHidd2Grads[offset], &pBiasOutGrads[offset], pFeaturesGrads);
    }
}

////////////////////////////////////////////////////////////////////////////////// CPU

void spatialConvCPU(
    bool pAvg,
    bool pScaleInv,
    int pNumNeighbors,
    int pNumInFeatures,
    int pNumOutFeatures,
    int pNumSamples,
    bool pCombin,
    float pRadius,
    const float* pInPoints,
    const int* pBatchIds,
    const float* pInFeatures,
    const float* pPDFs,
    const float* pSamples,
    const int* pStartIndexs,
    const int* pPackedNeighs,
    const float* pAABBMin,
    const float* pAABBMax,
    const float* pWeights1,
    const float* pBiases1,
    const float* pWeights2,
    const float* pBiases2,
    const float* pWeightsOut,
    const float* pBiasesOut,
    float* pOutFeatues)
{
#ifdef PRINT_CONV_INFO
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif
    
    //Evaluate MLP.
    if(pCombin){
        cudaMemset(pOutFeatues, 0, pNumOutFeatures*pNumSamples*sizeof(float));

        int numBlocksPerPoint = (pNumOutFeatures*pNumInFeatures)/BLOCK_MLP_SIZE;
        numBlocksPerPoint += ((pNumOutFeatures*pNumInFeatures)%BLOCK_MLP_SIZE != 0)?1:0;
        dim3 gridDimension = computeBlockGrid(
            (unsigned long long int)pNumNeighbors*
            (unsigned long long int)numBlocksPerPoint*
            (unsigned long long int)BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE);

        evaluateMLPKernel<<<gridDimension, EXECUTION_BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE*2*sizeof(float)>>>(
            pAvg, pScaleInv, pNumSamples, pNumNeighbors, pNumInFeatures, pNumOutFeatures, pRadius, pAABBMin, pAABBMax, 
            pWeights1, pWeights2, pWeightsOut, pBiases1, pBiases2, pBiasesOut, pSamples, pInPoints, pBatchIds, 
            pInFeatures, pStartIndexs, pPackedNeighs, pPDFs, pOutFeatues);

        gpuErrchk(cudaPeekAtLastError());
    }else{
        cudaMemset(pOutFeatues, 0, pNumInFeatures*pNumSamples*sizeof(float));

        int numBlocksPerPoint = (pNumInFeatures)/BLOCK_MLP_SIZE;
        numBlocksPerPoint += ((pNumInFeatures)%BLOCK_MLP_SIZE != 0)?1:0;
        dim3 gridDimension = computeBlockGrid(
            (unsigned long long int)pNumNeighbors*
            (unsigned long long int)numBlocksPerPoint*
            (unsigned long long int)BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE);

        evaluateMLPNoCombinKernel<<<gridDimension, EXECUTION_BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE*2*sizeof(float)>>>(
            pAvg, pScaleInv, pNumSamples, pNumNeighbors, pNumInFeatures, pRadius, pAABBMin, pAABBMax, 
            pWeights1, pWeights2, pWeightsOut, pBiases1, pBiases2, pBiasesOut, pSamples, pInPoints, pBatchIds, 
            pInFeatures, pStartIndexs, pPackedNeighs, pPDFs, pOutFeatues);

        gpuErrchk(cudaPeekAtLastError());
    }

#ifdef PRINT_CONV_INFO
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Forward Num points: %d | Neighbors: %d | Time %f\n", pNumSamples, pNumNeighbors, milliseconds);
#endif
}

void spatialConvGradsCPU(
    bool pAvg,
    bool pScaleInv,
    int pNumNeighbors,
    int pNumInFeatures,
    int pNumOutFeatures,
    int pNumSamples,
    int pNumPoints,
    bool pCombin,
    float pRadius,
    const float* pInPoints,
    const int* pBatchIds,
    const float* pInFeatures,
    const float* pPDFs,
    const float* pSamples,
    const int* pStartIndexs,
    const int* pPackedNeighs,
    const float* pAABBMin,
    const float* pAABBMax,
    const float* pWeights1,
    const float* pBiases1,
    const float* pWeights2,
    const float* pBiases2,
    const float* pWeightsOut,
    const float* pBiasesOut,
    const float* pInOutFeatueGrads,
    float* pOutFeatureGrads,
    float* pWeights1Grads,
    float* pWeight2Grads,
    float* pWeightOutGrads,
    float* pBiases1Grads,
    float* pBiases2Grads,
    float* pBiasesOutGrads)
{
#ifdef PRINT_CONV_INFO
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    //Compute dconv_j_d.
    if(pCombin){
        int numBlocksPerPoint = (pNumOutFeatures*pNumInFeatures)/BLOCK_MLP_SIZE;
        numBlocksPerPoint += ((pNumOutFeatures*pNumInFeatures)%BLOCK_MLP_SIZE != 0)?1:0;

        cudaMemset(pWeights1Grads, 0, sizeof(float)*3*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pWeight2Grads, 0, sizeof(float)*BLOCK_MLP_SIZE*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pWeightOutGrads, 0, sizeof(float)*(pNumOutFeatures*pNumInFeatures)*BLOCK_MLP_SIZE);
        cudaMemset(pBiases1Grads, 0, sizeof(float)*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pBiases2Grads, 0, sizeof(float)*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pBiasesOutGrads, 0, sizeof(float)*(pNumOutFeatures*pNumInFeatures));
        cudaMemset(pOutFeatureGrads, 0, sizeof(float)*pNumPoints*pNumInFeatures);
        
        dim3 gridDimension = computeBlockGrid(pNumNeighbors*numBlocksPerPoint*BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE);

        computedconvj_dKernel<<<gridDimension, EXECUTION_BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE*4*sizeof(float)>>>(
            pAvg, pScaleInv, pNumSamples, pNumNeighbors, pNumInFeatures, 
            pNumOutFeatures, pRadius, pAABBMin, pAABBMax, pWeights1, pWeights2, pWeightsOut, pBiases1, pBiases2, pBiasesOut, 
            pSamples, pInPoints, pBatchIds, pInFeatures, pInOutFeatueGrads, pStartIndexs, pPackedNeighs, pPDFs, pWeights1Grads, 
            pWeight2Grads, pWeightOutGrads, pBiases1Grads, pBiases2Grads, pBiasesOutGrads, pOutFeatureGrads);
        
        gpuErrchk(cudaPeekAtLastError());
    }else{
        int numBlocksPerPoint = (pNumInFeatures)/BLOCK_MLP_SIZE;
        numBlocksPerPoint += ((pNumInFeatures)%BLOCK_MLP_SIZE != 0)?1:0;

        cudaMemset(pWeights1Grads, 0, sizeof(float)*3*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pWeight2Grads, 0, sizeof(float)*BLOCK_MLP_SIZE*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pWeightOutGrads, 0, sizeof(float)*pNumInFeatures*BLOCK_MLP_SIZE);
        cudaMemset(pBiases1Grads, 0, sizeof(float)*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pBiases2Grads, 0, sizeof(float)*numBlocksPerPoint*BLOCK_MLP_SIZE);
        cudaMemset(pBiasesOutGrads, 0, sizeof(float)*pNumInFeatures);
        cudaMemset(pOutFeatureGrads, 0, sizeof(float)*pNumPoints*pNumInFeatures);
        
        dim3 gridDimension = computeBlockGrid(pNumNeighbors*numBlocksPerPoint*BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE);

        computedconvj_dNoCombinKernel<<<gridDimension, EXECUTION_BLOCK_MLP_SIZE, EXECUTION_BLOCK_MLP_SIZE*4*sizeof(float)>>>(
            pAvg, pScaleInv, pNumSamples, pNumNeighbors, pNumInFeatures, 
            pRadius, pAABBMin, pAABBMax, pWeights1, pWeights2, pWeightsOut, pBiases1, pBiases2, pBiasesOut, 
            pSamples, pInPoints, pBatchIds, pInFeatures, pInOutFeatueGrads, pStartIndexs, pPackedNeighs, pPDFs, pWeights1Grads, 
            pWeight2Grads, pWeightOutGrads, pBiases1Grads, pBiases2Grads, pBiasesOutGrads, pOutFeatureGrads);
        
        gpuErrchk(cudaPeekAtLastError());
    }

#ifdef PRINT_CONV_INFO
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Backward Num points: %d | Neighbors: %d | Time %f\n", pNumSamples, pNumNeighbors, milliseconds);
#endif
}
