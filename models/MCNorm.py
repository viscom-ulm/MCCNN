'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCNorm.py

    \brief Definition of the network architecture MCNorm for  
           normal estimation tasks.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import math
import sys
import os
import tensorflow as tf
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
from MCConvModule import compute_aabb, sort_features_back
from MCNetworkUtils import fully_connected, batch_norm_RELU_drop_out, distribute_grid, compute_convolution, \
    prepare_conv, sampling_and_conv_pooling, reduce_channels

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, isTraining):

    #Bounging box computation
    with tf.name_scope('aabbLayer'):
        aabbMin, aabbMax = compute_aabb(points, batchIds, batchSize)

    ############################################ Encoder

    # First Convolution
    sortPts1, sortBatchs1, sortFeatures1, cellIndexs1, indexs1 = distribute_grid('grid_1', points, batchIds, features, aabbMin, aabbMax, batchSize, 0.1)
    pdfs1, startIndexs1, packedNeighs1 = prepare_conv('pre_conv_1', sortPts1, sortBatchs1, sortPts1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, batchSize, 0.1, 0.25)
    convFeatures1 = compute_convolution("Conv_1", sortPts1, sortFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.1, 
        numInputFeatures, k, isTraining)
    
    # First Pooling
    resConvFeatures1 = batch_norm_RELU_drop_out("Conv_1_Reduce_BN", convFeatures1, isTraining, False, False)
    resConvFeatures1 = reduce_channels("Conv_1_Reduce", resConvFeatures1, k, k*2)
    sampledPts1, sampledBatchsIds1, sampledFeatures1 = sampling_and_conv_pooling("Sampling_and_pooling_1", sortPts1, resConvFeatures1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, 
        0.1, 0.2, k*2, isTraining, False, False, batchSize, 0.20)

    # Second Convolution
    sortPts2, sortBatchs2, sortFeatures2, cellIndexs2, _ = distribute_grid('Grid_2', sampledPts1, sampledBatchsIds1, sampledFeatures1, aabbMin, aabbMax, batchSize, 0.4)
    pdfs2, startIndexs2, packedNeighs2 = prepare_conv('pre_conv_2', sortPts2, sortBatchs2, sortPts2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, batchSize, 0.4, 0.25)
    exFeatures2 = batch_norm_RELU_drop_out("Conv_2_In", sortFeatures2, isTraining, False, False)
    convFeatures2 = compute_convolution("Conv_2", sortPts2, exFeatures2, sortBatchs2, pdfs2, sortPts2, startIndexs2, packedNeighs2, aabbMin, aabbMax, batchSize, 0.4, 
        k*2, k*2, isTraining, fullConv = False)
    convFeatures2 = tf.concat([sortFeatures2, convFeatures2], 1)
    
    # Second Pooling
    resConvFeatures2 = batch_norm_RELU_drop_out("Conv_2_Reduce_BN", convFeatures2, isTraining, False, False)
    resConvFeatures2 = reduce_channels("Conv_2_Reduce", resConvFeatures2, k*4, k*4)
    sampledPts2, sampledBatchsIds2, sampledFeatures2 = sampling_and_conv_pooling("Sampling_and_pooling_2", sortPts2, resConvFeatures2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, 
        0.4, 0.8, k*4, isTraining, False, False, batchSize, 0.20)
    
    # Third Convolution
    sortPts3, sortBatchs3, sortFeatures3, cellIndexs3, _ = distribute_grid('Grid_3', sampledPts2, sampledBatchsIds2, sampledFeatures2, aabbMin, aabbMax, batchSize, math.sqrt(3))
    pdfs3, startIndexs3, packedNeighs3 = prepare_conv('pre_conv_3', sortPts3, sortBatchs3, sortPts3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, batchSize, math.sqrt(3), 0.25)
    exFeatures3 = batch_norm_RELU_drop_out("Conv_3_In", sortFeatures3, isTraining, False, False)
    convFeatures3 = compute_convolution("Conv_3", sortPts3, exFeatures3, sortBatchs3, pdfs3, sortPts3, startIndexs3, packedNeighs3, aabbMin, aabbMax, batchSize, math.sqrt(3), 
        k*4, k*4, isTraining, fullConv = False)
    convFeatures3 = tf.concat([sortFeatures3, convFeatures3], 1)

    
    #####################################################  Decoder
    
    # Second upsampling
    resConvFeatures2_3 = batch_norm_RELU_drop_out("Up_2_3_BN", convFeatures3, isTraining, False, False)
    pdfs2_3, startIndexs2_3, packedNeighs2_3 = prepare_conv('Pre_Up_2_3', sortPts2, sortBatchs2, sortPts3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, batchSize, 0.8, 0.25)
    convFeatures2_3 = compute_convolution("Up_2_3", sortPts3, resConvFeatures2_3, sortBatchs3, pdfs2_3, sortPts2, startIndexs2_3, packedNeighs2_3, aabbMin, aabbMax, batchSize, 
        0.8, k*8, k*8, isTraining, fullConv = False)
    upFeatures2_3 = tf.concat([convFeatures2_3, convFeatures2], 1)
    upFeatures2_3 = batch_norm_RELU_drop_out("Ur_red_2_3", upFeatures2_3, isTraining, False, False)
    upFeatures2_3 = reduce_channels("Conv_2_3_Reduce", upFeatures2_3, k*12, k*4)
    upFeatures2_3 = batch_norm_RELU_drop_out("DeConv_BN_2_3", upFeatures2_3, isTraining, False, False)
    upFeatures2_3 = compute_convolution("DeConv_2_3", sortPts2, upFeatures2_3, sortBatchs2, pdfs2, sortPts2, startIndexs2, packedNeighs2, aabbMin, aabbMax, batchSize, 0.4, 
        k*4, k*4, isTraining, fullConv = False)
    
    # First upsampling
    resConvFeatures1_2 = batch_norm_RELU_drop_out("Up_1_2_BN", upFeatures2_3, isTraining, False, False)
    pdfs1_2, startIndexs1_2, packedNeighs1_2 = prepare_conv('Pre_Up_1_2', sortPts1, sortBatchs1, sortPts2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, batchSize, 0.2, 0.25)
    convFeatures1_2 = compute_convolution("Up_1_2", sortPts2, resConvFeatures1_2, sortBatchs2, pdfs1_2, sortPts1, startIndexs1_2, packedNeighs1_2, aabbMin, aabbMax, batchSize, 
        0.2, k*4, k*4, isTraining, fullConv = False)
    upFeatures1 = tf.concat([convFeatures1_2, convFeatures1], 1)
    upFeatures1 = batch_norm_RELU_drop_out("Ur_red_1", upFeatures1, isTraining, False, False)
    upFeatures1 = reduce_channels("Conv_1_Reduce", upFeatures1, k*5, k*2)
    upFeatures1 = batch_norm_RELU_drop_out("DeConv_BN_1", upFeatures1, isTraining, False, False)
    normals = compute_convolution("DeConv_1", sortPts1, upFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.1, 
        k*2, 3, isTraining)
    
    normals = sort_features_back(normals, indexs1)

    return normals
