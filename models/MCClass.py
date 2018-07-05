'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCClass.py

    \brief Definition of the network architecture MCClass for classification 
           tasks.

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
    prepare_conv, sampling_and_conv_pooling, sampling_center_and_conv_pooling, reduce_channels

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, numOutCat, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    #Bounging box computation
    with tf.name_scope('aabbLayer'):
        aabbMin, aabbMax = compute_aabb(points, batchIds, batchSize)

    ############################################ Hierarchy traversal

    # First Convolution
    sortPts1, sortBatchs1, sortFeatures1, cellIndexs1, _ = distribute_grid('grid_1', points, batchIds, features, aabbMin, aabbMax, batchSize, 0.1)
    pdfs1, startIndexs1, packedNeighs1 = prepare_conv('pre_conv_1', sortPts1, sortBatchs1, sortPts1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, batchSize, 0.1, 0.25)
    convFeatures1 = compute_convolution("Conv_1", sortPts1, sortFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.1, 
        numInputFeatures, k, isTraining)
    
    # First Pooling
    resConvFeatures1 = batch_norm_RELU_drop_out("Conv_1_Reduce_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    resConvFeatures1 = reduce_channels("Conv_1_Reduce", resConvFeatures1, k, k*2)
    sampledPts1, sampledBatchsIds1, sampledFeatures1 = sampling_and_conv_pooling("Sampling_and_pooling_1", sortPts1, resConvFeatures1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, 
        0.1, 0.2, k*2, isTraining, useConvDropOut, keepProbConv, batchSize, 0.20)

    # Second Convolution
    sortPts2, sortBatchs2, sortFeatures2, cellIndexs2, _ = distribute_grid('Grid_2', sampledPts1, sampledBatchsIds1, sampledFeatures1, aabbMin, aabbMax, batchSize, 0.4)
    pdfs2, startIndexs2, packedNeighs2 = prepare_conv('pre_conv_2', sortPts2, sortBatchs2, sortPts2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, batchSize, 0.4, 0.25)
    exFeatures2 = batch_norm_RELU_drop_out("Conv_2_In", sortFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = compute_convolution("Conv_2", sortPts2, exFeatures2, sortBatchs2, pdfs2, sortPts2, startIndexs2, packedNeighs2, aabbMin, aabbMax, batchSize, 0.4, 
        k*2, k*2, isTraining, fullConv = False)
    convFeatures2 = tf.concat([sortFeatures2, convFeatures2], 1)
    
    # Second Pooling
    resConvFeatures2 = batch_norm_RELU_drop_out("Conv_2_Reduce_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    resConvFeatures2 = reduce_channels("Conv_2_Reduce", resConvFeatures2, k*4, k*8)
    sampledPts2, sampledBatchsIds2, sampledFeatures2 = sampling_and_conv_pooling("Sampling_and_pooling_2", sortPts2, resConvFeatures2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, 
        0.4, 0.8, k*8, isTraining, useConvDropOut, keepProbConv, batchSize, 0.20)
    
    # Third Convolution
    sortPts3, sortBatchs3, sortFeatures3, cellIndexs3, _ = distribute_grid('Grid_3', sampledPts2, sampledBatchsIds2, sampledFeatures2, aabbMin, aabbMax, batchSize, 1.1)
    pdfs3, startIndexs3, packedNeighs3 = prepare_conv('pre_conv_3', sortPts3, sortBatchs3, sortPts3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, batchSize, 1.1, 0.25)
    exFeatures3 = batch_norm_RELU_drop_out("Conv_3_In", sortFeatures3, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = compute_convolution("Conv_3", sortPts3, exFeatures3, sortBatchs3, pdfs3, sortPts3, startIndexs3, packedNeighs3, aabbMin, aabbMax, batchSize, 1.1, 
        k*8, k*8, isTraining, fullConv = False)
    convFeatures3 = tf.concat([sortFeatures3, convFeatures3], 1)

    # Third Pooling
    resConvFeatures3 = batch_norm_RELU_drop_out("Conv_3_Reduce_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    resConvFeatures3 = reduce_channels("Conv_3_Reduce", resConvFeatures3, k*16, k*32)
    _, _, sampledFeatures3 = sampling_center_and_conv_pooling("Sampling_and_pooling_3", sortPts3, resConvFeatures3, sortBatchs3, aabbMin, aabbMax, 
        k*32, isTraining, useConvDropOut, keepProbConv, batchSize, 0.2)
        
    #Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_final", sampledFeatures3, isTraining, useConvDropOut, keepProbConv)
    finalLogits = fully_connected(finalInput, k*32, k*16, k*8, numOutCat, "Final_Logits", keepProbFull, isTraining, useDropOutFull)

    return finalLogits
