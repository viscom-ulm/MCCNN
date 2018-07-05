'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCClassH.py

    \brief Definition of the network architecture MCClassH for classification 
           tasks, in which the class probabilities are computed by two 
           separated paths.

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
    prepare_conv, reduce_channels, compute_hierarchy

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

    ############################################ LOGITS 1

    ############################################ First level convolutions

    #### Convolutions
    currPts = points
    currBatchIds = batchIds

    currRadius = 0.1
    currSampledPts = sampledPts[0]
    currSampledBatchIds = sampledBatchIds[0]
    currSampledFeatures = sampledFeatures[0]

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_1', currSampledPts, 
        currSampledBatchIds, currSampledFeatures, aabbMin, aabbMax, batchSize, currRadius)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_conv_1', currPts, currBatchIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, currRadius, 0.25)
    currFeatures = compute_convolution("Conv_1", sortPts, sortFeatures, sortBatchs, pdfs, currPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, currRadius, numInputFeatures, k, isTraining)


    ############################################ Second level convolutions

    #### Pooling
    newCurrPts = sampledPts[1]
    newCurrBatchsIds = sampledBatchIds[1]
    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_pool_1', currPts, 
        currBatchIds, currFeatures, aabbMin, aabbMax, batchSize, 0.2)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_pool_1', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, 0.2, 0.20)
    sortFeatures = batch_norm_RELU_drop_out("Pool_1_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("Pool_1", sortFeatures, k, k*2)
    sortFeatures = batch_norm_RELU_drop_out("Pool_1_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    newCurrFeatures = compute_convolution("Pool_1", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, 0.2, k*2, k*2, isTraining, fullConv = False)

    #### Convolutions
    currRadius = 0.4
    currSampledPts = sampledPts[1]
    currSampledBatchIds = sampledBatchIds[1]
    currSampledFeatures = newCurrFeatures

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_2', currSampledPts, 
        currSampledBatchIds, currSampledFeatures, aabbMin, aabbMax, batchSize, currRadius)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_conv_2', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, currRadius, 0.25)
    sortFeatures = batch_norm_RELU_drop_out("Conv_2_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("Conv_2_Reduce", sortFeatures, k*2, k*2)
    sortFeatures = batch_norm_RELU_drop_out("Conv_2_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    currFeatures = compute_convolution("Conv_2", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, currRadius, k*2, k*2, isTraining, fullConv = False)

    
    currPts = newCurrPts
    currBatchIds = newCurrBatchsIds
    currFeatures = tf.concat([currFeatures, newCurrFeatures], 1)

    ############################################ Third level convolutions

    #### Pooling
    newCurrPts = sampledPts[2]
    newCurrBatchsIds = sampledBatchIds[2]
    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_pool_2', currPts, 
        currBatchIds, currFeatures, aabbMin, aabbMax, batchSize, 0.8)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_pool_2', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, 0.8, 0.20)
    sortFeatures = batch_norm_RELU_drop_out("Pool_2_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("Pool_2", sortFeatures, k*4, k*8)
    sortFeatures = batch_norm_RELU_drop_out("Pool_2_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    newCurrFeatures = compute_convolution("Pool_2", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, 0.8, k*8, k*8, isTraining, fullConv = False)

    #### Compute hierarchy
    newSampledFeatures = newCurrFeatures

    #### Convolutions
    currRadius = 1.2
    currSampledPts = sampledPts[2]
    currSampledBatchIds = sampledBatchIds[2]
    currSampledFeatures = newSampledFeatures

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_3', currSampledPts, 
        currSampledBatchIds, currSampledFeatures, aabbMin, aabbMax, batchSize, currRadius)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_conv_3', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, currRadius, 0.25)
    sortFeatures = batch_norm_RELU_drop_out("Conv_3_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("Conv_3_Reduce", sortFeatures, k*8, k*8)
    sortFeatures = batch_norm_RELU_drop_out("Conv_3_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    currFeatures = compute_convolution("Conv_3", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, currRadius, k*8, k*8, isTraining, fullConv = False)
            
    currPts = newCurrPts
    currBatchIds = newCurrBatchsIds
    currFeatures = tf.concat([currFeatures, newCurrFeatures], 1)

    ############################################ Fourth level convolutions

    #### Pooling
    newCurrPts = sampledPts[3]
    newCurrBatchsIds = sampledBatchIds[3]
    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('grid_pool_3', currPts, 
        currBatchIds, currFeatures, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1)
    pdfs, startIndexs, packedNeighs = prepare_conv('pre_pool_3', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, 0.20)
    sortFeatures = batch_norm_RELU_drop_out("Pool_3_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("Pool_3", sortFeatures, k*16, k*32)
    sortFeatures = batch_norm_RELU_drop_out("Pool_3_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    newCurrFeatures = compute_convolution("Pool_3", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, k*32, k*32, isTraining, fullConv = False)

    
    #Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_final", newCurrFeatures, isTraining, useConvDropOut, keepProbConv)
    finalLogits1 = fully_connected(finalInput, k*32, k*16, k*8, numOutCat, "Final_Logits", keepProbFull, isTraining, useDropOutFull)


    ############################################ LOGITS 2

    ############################################ Second level convolutions

    #### Pooling
    newCurrPts = sampledPts[1]
    newCurrBatchsIds = sampledBatchIds[1]
    newCurrFeatures = sampledFeatures[1]

    #### Convolutions
    currRadius = 0.4
    currSampledPts = sampledPts[1]
    currSampledBatchIds = sampledBatchIds[1]
    currSampledFeatures = newCurrFeatures

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('2grid_2', currSampledPts, 
        currSampledBatchIds, currSampledFeatures, aabbMin, aabbMax, batchSize, currRadius)
    pdfs, startIndexs, packedNeighs = prepare_conv('2pre_conv_2', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, currRadius, 0.25)
    currFeatures = compute_convolution("2Conv_2", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, currRadius, 1, k*2, isTraining)

    currPts = newCurrPts
    currBatchIds = newCurrBatchsIds

    ############################################ Third level convolutions

    #### Pooling
    newCurrPts = sampledPts[2]
    newCurrBatchsIds = sampledBatchIds[2]
    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('2grid_pool_2', currPts, 
        currBatchIds, currFeatures, aabbMin, aabbMax, batchSize, 0.8)
    pdfs, startIndexs, packedNeighs = prepare_conv('2pre_pool_2', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, 0.8, 0.20)
    sortFeatures = batch_norm_RELU_drop_out("2Pool_2_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("2Pool_2", sortFeatures, k*2, k*8)
    sortFeatures = batch_norm_RELU_drop_out("2Pool_2_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    newCurrFeatures = compute_convolution("2Pool_2", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, 0.8, k*8, k*8, isTraining, fullConv = False)

    #### Compute hierarchy
    newSampledFeatures = newCurrFeatures

    #### Convolutions
    currRadius = 1.2
    currSampledPts = sampledPts[2]
    currSampledBatchIds = sampledBatchIds[2]
    currSampledFeatures = newSampledFeatures

    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('2grid_3', currSampledPts, 
        currSampledBatchIds, currSampledFeatures, aabbMin, aabbMax, batchSize, currRadius)
    pdfs, startIndexs, packedNeighs = prepare_conv('2pre_conv_3', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, currRadius, 0.25)
    sortFeatures = batch_norm_RELU_drop_out("2Conv_3_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("2Conv_3_Reduce", sortFeatures, k*8, k*8)
    sortFeatures = batch_norm_RELU_drop_out("2Conv_3_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    currFeatures = compute_convolution("2Conv_3", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, currRadius, k*8, k*8, isTraining, fullConv = False)
            
    currPts = newCurrPts
    currBatchIds = newCurrBatchsIds
    currFeatures = tf.concat([currFeatures, newCurrFeatures], 1)

    ############################################ Fourth level convolutions

    #### Pooling
    newCurrPts = sampledPts[3]
    newCurrBatchsIds = sampledBatchIds[3]
    sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid('2grid_pool_3', currPts, 
        currBatchIds, currFeatures, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1)
    pdfs, startIndexs, packedNeighs = prepare_conv('2pre_pool_3', newCurrPts, newCurrBatchsIds, sortPts, sortBatchs, 
        cellIndexs, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, 0.20)
    sortFeatures = batch_norm_RELU_drop_out("2Pool_3_Reduce_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    sortFeatures = reduce_channels("2Pool_3", sortFeatures, k*16, k*32)
    sortFeatures = batch_norm_RELU_drop_out("2Pool_3_BN", sortFeatures, isTraining, useConvDropOut, keepProbConv)
    newCurrFeatures = compute_convolution("2Pool_3", sortPts, sortFeatures, sortBatchs, pdfs, newCurrPts, startIndexs, 
            packedNeighs, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, k*32, k*32, isTraining, fullConv = False)

    #Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("2BNRELUDROP_final", newCurrFeatures, isTraining, useConvDropOut, keepProbConv)
    finalLogits2 = fully_connected(finalInput, k*32, k*16, k*8, numOutCat, "2Final_Logits", keepProbFull, isTraining, useDropOutFull)

    ############################################ PATH DROPOUT
    counter = tf.constant(0.0, dtype=tf.float32)

    probability = tf.random_uniform([1])

    mask1 = tf.less_equal(probability[0], tf.constant(0.66))
    mask1 = tf.maximum(tf.cast(mask1, tf.float32), tf.cast(tf.logical_not(isTraining), tf.float32))
    counter = tf.add(counter, mask1)
    finalLogits1 = tf.scalar_mul(mask1, finalLogits1)

    mask2 = tf.greater_equal(probability[0], tf.constant(0.33))
    mask2 = tf.maximum(tf.cast(mask2, tf.float32), tf.cast(tf.logical_not(isTraining), tf.float32))
    counter = tf.add(counter, mask2)
    finalLogits2 = tf.scalar_mul(mask2, finalLogits2)
    
    counter = tf.multiply(tf.constant((2.0), dtype=tf.float32), tf.reciprocal(counter))
    
    return tf.scalar_mul(counter, tf.add(finalLogits1, finalLogits2))
