'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCSeg.py

    \brief Definition of the network architecture MCSeg for  
           segmentation tasks.

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

def create_network(points, batchIds, features, catLabels, numInputFeatures, numCats, numParts, batchSize, k, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    #Bounging box computation
    with tf.name_scope('aabbLayer'):
        aabbMin, aabbMax = compute_aabb(points, batchIds, batchSize)

    ############################################ First hierarchy traversal

    # First Convolution
    sortPts1, sortBatchs1, sortFeatures1, cellIndexs1, indexs1 = distribute_grid('grid_1', points, batchIds, features, aabbMin, aabbMax, batchSize, 0.03)
    pdfs1, startIndexs1, packedNeighs1 = prepare_conv('pre_conv_1', sortPts1, sortBatchs1, sortPts1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, batchSize, 0.03, 0.25)
    convFeatures1 = compute_convolution("Conv_1", sortPts1, sortFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.03, 
        numInputFeatures, k, isTraining)
    
    # First Pooling
    resConvFeatures1 = batch_norm_RELU_drop_out("Conv_1_Reduce_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    resConvFeatures1 = reduce_channels("Conv_1_Reduce", resConvFeatures1, k, k*2)
    sampledPts1, sampledBatchsIds1, sampledFeatures1 = sampling_and_conv_pooling("Sampling_and_pooling_1", sortPts1, resConvFeatures1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, 
        0.025, 0.05, k*2, isTraining, useConvDropOut, keepProbConv, batchSize, 0.20)

    # Second Convolution
    sortPts2, sortBatchs2, sortFeatures2, cellIndexs2, _ = distribute_grid('Grid_2', sampledPts1, sampledBatchsIds1, sampledFeatures1, aabbMin, aabbMax, batchSize, 0.1)
    pdfs2, startIndexs2, packedNeighs2 = prepare_conv('pre_conv_2', sortPts2, sortBatchs2, sortPts2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, batchSize, 0.1, 0.25)
    exFeatures2 = batch_norm_RELU_drop_out("Conv_2_In", sortFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = compute_convolution("Conv_2", sortPts2, exFeatures2, sortBatchs2, pdfs2, sortPts2, startIndexs2, packedNeighs2, aabbMin, aabbMax, batchSize, 0.1, 
        k*2, k*2, isTraining, fullConv = False)
    convFeatures2 = tf.concat([sortFeatures2, convFeatures2], 1)
    
    # Second Pooling
    resConvFeatures2 = batch_norm_RELU_drop_out("Conv_2_Reduce_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    resConvFeatures2 = reduce_channels("Conv_2_Reduce", resConvFeatures2, k*4, k*4)
    sampledPts2, sampledBatchsIds2, sampledFeatures2 = sampling_and_conv_pooling("Sampling_and_pooling_2", sortPts2, resConvFeatures2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, 
        0.1, 0.2, k*4, isTraining, useConvDropOut, keepProbConv, batchSize, 0.20)
    
    # Third Convolution
    sortPts3, sortBatchs3, sortFeatures3, cellIndexs3, _ = distribute_grid('Grid_3', sampledPts2, sampledBatchsIds2, sampledFeatures2, aabbMin, aabbMax, batchSize, 0.4)
    pdfs3, startIndexs3, packedNeighs3 = prepare_conv('pre_conv_3', sortPts3, sortBatchs3, sortPts3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, batchSize, 0.4, 0.25)
    exFeatures3 = batch_norm_RELU_drop_out("Conv_3_In", sortFeatures3, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = compute_convolution("Conv_3", sortPts3, exFeatures3, sortBatchs3, pdfs3, sortPts3, startIndexs3, packedNeighs3, aabbMin, aabbMax, batchSize, 0.4, 
        k*4, k*4, isTraining, fullConv = False)
    convFeatures3 = tf.concat([sortFeatures3, convFeatures3], 1)

    # Third Pooling
    resConvFeatures3 = batch_norm_RELU_drop_out("Conv_3_Reduce_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    resConvFeatures3 = reduce_channels("Conv_3_Reduce", resConvFeatures3, k*8, k*8)
    sampledPts3, sampledBatchsIds3, sampledFeatures3 = sampling_and_conv_pooling("Sampling_and_pooling_3", sortPts3, resConvFeatures3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, 
        0.4, 0.8, k*8, isTraining, useConvDropOut, keepProbConv, batchSize, 0.20)
    
    # Fourth Convolution
    sortPts4, sortBatchs4, sortFeatures4, cellIndexs4, _ = distribute_grid('Grid_4', sampledPts3, sampledBatchsIds3, sampledFeatures3, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1)
    pdfs4, startIndexs4, packedNeighs4 = prepare_conv('pre_conv_4', sortPts4, sortBatchs4, sortPts4, sortBatchs4, cellIndexs4, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, 0.25)
    exFeatures4 = batch_norm_RELU_drop_out("Conv_4_In", sortFeatures4, isTraining, useConvDropOut, keepProbConv)
    convFeatures4 = compute_convolution("Conv_4", sortPts4, exFeatures4, sortBatchs4, pdfs4, sortPts4, startIndexs4, packedNeighs4, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, 
        k*8, k*8, isTraining, fullConv = False)
    convFeatures4 = tf.concat([sortFeatures4, convFeatures4], 1)

    
    ##################################################### Multi-hierarchy sampling
    
    # Third upsampling
    resConvFeatures3_4 = batch_norm_RELU_drop_out("Up_3_4_BN", convFeatures4, isTraining, useConvDropOut, keepProbConv)
    pdfs3_4, startIndexs3_4, packedNeighs3_4 = prepare_conv('Pre_Up_3_4', sortPts3, sortBatchs3, sortPts4, sortBatchs4, cellIndexs4, aabbMin, aabbMax, batchSize, 
        math.sqrt(3.0)+0.1, 0.25)
    convFeatures3_4 = compute_convolution("Up_3_4", sortPts4, resConvFeatures3_4, sortBatchs4, pdfs3_4, sortPts3, startIndexs3_4, packedNeighs3_4, aabbMin, aabbMax, batchSize, 
        math.sqrt(3.0)+0.1, k*16, k*16, isTraining, fullConv = False)
    upFeatures3_4 = tf.concat([convFeatures3_4, convFeatures3], 1)
    upFeatures3_4 = batch_norm_RELU_drop_out("Up_red_3_4", upFeatures3_4, isTraining, useConvDropOut, keepProbConv)
    upFeatures3_4 = reduce_channels("Up_3_4_Reduce", upFeatures3_4, k*24, k*8)
    upFeatures3_4 = batch_norm_RELU_drop_out("DeConv_BN_3_4", upFeatures3_4, isTraining, useConvDropOut, keepProbConv)
    upFeatures3_4 = compute_convolution("DeConv_3_4", sortPts3, upFeatures3_4, sortBatchs3, pdfs3, sortPts3, startIndexs3, packedNeighs3, aabbMin, aabbMax, batchSize, 0.4, 
        k*8, k*8, isTraining, fullConv = False)
    
    
    # Second upsampling
    resConvFeatures2_3 = batch_norm_RELU_drop_out("Up_2_3_BN", upFeatures3_4, isTraining, useConvDropOut, keepProbConv)
    pdfs2_3, startIndexs2_3, packedNeighs2_3 = prepare_conv('Pre_Up_2_3', sortPts2, sortBatchs2, sortPts3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, batchSize, 0.2, 0.25)
    convFeatures2_3 = compute_convolution("Up_2_3", sortPts3, resConvFeatures2_3, sortBatchs3, pdfs2_3, sortPts2, startIndexs2_3, packedNeighs2_3, aabbMin, aabbMax, batchSize, 
        0.2, k*8, k*8, isTraining, fullConv = False)
    upFeatures2_3 = tf.concat([convFeatures2_3, convFeatures2], 1)
    upFeatures2_3 = batch_norm_RELU_drop_out("Ur_red_2_3", upFeatures2_3, isTraining, useConvDropOut, keepProbConv)
    upFeatures2_3 = reduce_channels("Conv_2_3_Reduce", upFeatures2_3, k*12, k*4)
    upFeatures2_3 = batch_norm_RELU_drop_out("DeConv_BN_2_3", upFeatures2_3, isTraining, useConvDropOut, keepProbConv)
    upFeatures2_3 = compute_convolution("DeConv_2_3", sortPts2, upFeatures2_3, sortBatchs2, pdfs2, sortPts2, startIndexs2, packedNeighs2, aabbMin, aabbMax, batchSize, 0.1, 
        k*4, k*4, isTraining, fullConv = False)
    
    
    # First upsampling
    resConvFeatures1_2 = batch_norm_RELU_drop_out("Up_1_2_BN", upFeatures2_3, isTraining, useConvDropOut, keepProbConv)
    pdfs1_2, startIndexs1_2, packedNeighs1_2 = prepare_conv('Pre_Up_1_2', sortPts1, sortBatchs1, sortPts2, sortBatchs2, cellIndexs2, aabbMin, aabbMax, batchSize, 0.05, 0.25)
    convFeatures1_2 = compute_convolution("Up_1_2", sortPts2, resConvFeatures1_2, sortBatchs2, pdfs1_2, sortPts1, startIndexs1_2, packedNeighs1_2, aabbMin, aabbMax, batchSize, 
        0.05, k*4, k*4, isTraining, fullConv = False)
    resConvFeatures1_3 = batch_norm_RELU_drop_out("Up_1_3_BN", upFeatures3_4, isTraining, useConvDropOut, keepProbConv)
    pdfs1_3, startIndexs1_3, packedNeighs1_3 = prepare_conv('Pre_Up_1_3', sortPts1, sortBatchs1, sortPts3, sortBatchs3, cellIndexs3, aabbMin, aabbMax, batchSize, 0.2, 0.25)
    convFeatures1_3 = compute_convolution("Up_1_3", sortPts3, resConvFeatures1_3, sortBatchs3, pdfs1_3, sortPts1, startIndexs1_3, packedNeighs1_3, aabbMin, aabbMax, batchSize, 
        0.2, k*8, k*8, isTraining, fullConv = False)
    upFeatures1 = tf.concat([convFeatures1_2, convFeatures1_3, convFeatures1], 1)
    upFeatures1 = batch_norm_RELU_drop_out("Ur_red_1", upFeatures1, isTraining, useConvDropOut, keepProbConv)
    upFeatures1 = reduce_channels("Conv_1_Reduce", upFeatures1, k*13, k*4)
    upFeatures1 = batch_norm_RELU_drop_out("DeConv_BN_1", upFeatures1, isTraining, useConvDropOut, keepProbConv)
    upFeatures1 = compute_convolution("DeConv_1", sortPts1, upFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.03, 
        k*4, k*4, isTraining, fullConv = False)
    
    
    # Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_hier_final", upFeatures1, isTraining, useConvDropOut, keepProbConv)
    #Convert cat labels
    catLabelOneHot = tf.one_hot(catLabels, numCats, on_value=1.0, off_value=0.0)
    catLabelOneHot = tf.reshape(catLabelOneHot, [-1, numCats])
    finalInput = tf.concat([catLabelOneHot, finalInput], 1)
    finalLogits = fully_connected(finalInput, k*4 + numCats, k*4, k*2, numParts, "Final_Logits", keepProbFull, isTraining, useDropOutFull, useInitBN = False)
    finalLogits = sort_features_back(finalLogits, indexs1)

    return finalLogits
