'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCNormSmall.py

    \brief Definition of the network architecture MCNormSmall for  
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
from MCNetworkUtils import fully_connected, batch_norm_RELU_drop_out, distribute_grid, compute_convolution, prepare_conv

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, isTraining, multiConv = True, useMC = True):

    #Bounging box computation
    with tf.name_scope('aabbLayer'):
        aabbMin, aabbMax = compute_aabb(points, batchIds, batchSize)

    #Distribute points regular grid
    sortPts1, sortBatchs1, sortFeatures1, cellIndexs1, indexs1 = distribute_grid('grid_1', points, batchIds, features, aabbMin, aabbMax, batchSize, 0.15)
    
    #Prepare convolutions
    pdfs1, startIndexs1, packedNeighs1 = prepare_conv('pre_conv_1', sortPts1, sortBatchs1, sortPts1, sortBatchs1, cellIndexs1, aabbMin, aabbMax, batchSize, 0.15, 0.2)
    
    #Convolution 1
    convFeatures1 = compute_convolution("Conv_1", sortPts1, sortFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.15, numInputFeatures, k, isTraining)

    #BatchNorm and RELU
    convFeatures1 = batch_norm_RELU_drop_out("BN_RELU", convFeatures1, isTraining, False, False)
    
    #Convolution 2
    normals = compute_convolution("Conv_2", sortPts1, convFeatures1, sortBatchs1, pdfs1, sortPts1, startIndexs1, packedNeighs1, aabbMin, aabbMax, batchSize, 0.15, k, 3, isTraining)

    #Sort normals to original position
    normals = sort_features_back(normals, indexs1)

    return normals
