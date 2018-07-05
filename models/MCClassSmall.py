'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCClassSmall.py

    \brief Definition of the network architecture MCClassSmall for  
           classification tasks.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
from MCConvModule import compute_aabb
from MCNetworkUtils import fully_connected, batch_norm_RELU_drop_out, distribute_grid, compute_convolution, \
    prepare_conv, sampling_and_conv_pooling, reduce_channels, compute_hierarchy

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, numOutCat, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    #Bounging box computation
    with tf.name_scope('aabbLayer'):
        aabbMin, aabbMax = compute_aabb(points, batchIds, batchSize)

    #### Compute hierarchy
    samplingRadius = [0.1, 0.4, math.sqrt(3.0)+0.1]
    sampledPts, sampledBatchIds, sampledFeatures, _ = compute_hierarchy(
        "Hierarchy", points, batchIds, features, samplingRadius, aabbMin, aabbMax, batchSize)

    centerPts = (aabbMax + aabbMin)*0.5
    sampledPts[3] = tf.slice(centerPts, [0, 0], tf.shape(sampledPts[3]))

    ############################################ Convolutions

    #### Convolution 1
    srcConvPts = sampledPts[0]
    srcConvBatchs = sampledBatchIds[0]
    srcConvFeatures = sampledFeatures[0]

    destConvPts = sampledPts[1]
    destConvBatchs = sampledBatchIds[1]

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_1', srcConvPts, srcConvBatchs, srcConvFeatures, aabbMin, aabbMax, batchSize, 0.2)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_conv_1', destConvPts, destConvBatchs, sortPts, sortBatchs, cellIndexs, aabbMin, aabbMax, batchSize, 0.2, 0.2)
    destConvFeatures = compute_convolution("Conv_1", sortPts, sortFeatures, sortBatchs, pdfs, destConvPts, startIndexs, packedNeighs, aabbMin, aabbMax, batchSize, 0.2, numInputFeatures, k, isTraining)

    #### Convolution 2
    srcConvPts = destConvPts
    srcConvBatchs = destConvBatchs
    srcConvFeatures = destConvFeatures

    destConvPts = sampledPts[2]
    destConvBatchs = sampledBatchIds[2]

    srcConvFeatures = batch_norm_RELU_drop_out("Pool_1_Reduce_BN", srcConvFeatures, isTraining, useConvDropOut, keepProbConv)
    srcConvFeatures = reduce_channels("Pool_1", srcConvFeatures, k, k*2)
    srcConvFeatures = batch_norm_RELU_drop_out("Pool_1_2_Reduce_BN", srcConvFeatures, isTraining, useConvDropOut, keepProbConv)

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_2', srcConvPts, srcConvBatchs, srcConvFeatures, aabbMin, aabbMax, batchSize, 0.8)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_conv_2', destConvPts, destConvBatchs, sortPts, sortBatchs, cellIndexs, aabbMin, aabbMax, batchSize, 0.8, 0.2)
    destConvFeatures = compute_convolution("Conv_2", sortPts, sortFeatures, sortBatchs, pdfs, destConvPts, startIndexs, packedNeighs, aabbMin, aabbMax, batchSize, 0.8, k*2, k*2, isTraining, fullConv = False)

    #### Convolution 3
    srcConvPts = destConvPts
    srcConvBatchs = destConvBatchs
    srcConvFeatures = destConvFeatures

    destConvPts = sampledPts[3]
    destConvBatchs = sampledBatchIds[3]

    srcConvFeatures = batch_norm_RELU_drop_out("Pool_2_Reduce_BN", srcConvFeatures, isTraining, useConvDropOut, keepProbConv)
    srcConvFeatures = reduce_channels("Pool_2", srcConvFeatures, k*2, k*4)
    srcConvFeatures = batch_norm_RELU_drop_out("Pool_2_2_Reduce_BN", srcConvFeatures, isTraining, useConvDropOut, keepProbConv)

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_3', srcConvPts, srcConvBatchs, srcConvFeatures, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_conv_3', destConvPts, destConvBatchs, sortPts, sortBatchs, cellIndexs, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, 0.2)
    destConvFeatures = compute_convolution("Conv_3", sortPts, sortFeatures, sortBatchs, pdfs, destConvPts, startIndexs, packedNeighs, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, k*4, k*4, isTraining, fullConv = False)

    
    #Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_final", destConvFeatures, isTraining, useConvDropOut, keepProbConv)
    finalLogits = fully_connected(finalInput, k*4, k*2, k, numOutCat, "Final_Logits", keepProbFull, isTraining, useDropOutFull)

    return finalLogits
