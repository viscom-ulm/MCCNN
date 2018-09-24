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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import batch_norm_RELU_drop_out, conv_1x1

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, isTraining, multiConv = True, useMC = True):

    ############################################  Compute point hierarchy
    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.1, 0.4], "MCNorm_PH", batchSize)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25)

    ############################################ Encoder

    # First Convolution
    convFeatures1 = mConvBuilder.create_convolution(
        convName="Conv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k,
        convRadius=0.1,
        multiFeatureConv=True)
    
    # First Pooling
    bnConvFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_In_BN", convFeatures1, isTraining, False, False)
    bnConvFeatures1 = conv_1x1("Reduce_Pool_1", bnConvFeatures1, k, k*2)
    bnConvFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_Out_BN", bnConvFeatures1, isTraining, False, False)
    poolFeatures1 = mConvBuilder.create_convolution(
        convName="Pool_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        outPointLevel=1, 
        inFeatures=bnConvFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.2,
        KDEWindow= 0.2)

    # Second Convolution
    bnPoolFeatures1 = batch_norm_RELU_drop_out("Conv_2_In_BN", poolFeatures1, isTraining, False, False)
    convFeatures2 = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=bnPoolFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.4)
    convFeatures2 = tf.concat([poolFeatures1, convFeatures2], 1)
    
    # Second Pooling
    bnConvFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_In_BN", convFeatures2, isTraining, False, False)
    bnConvFeatures2 = conv_1x1("Reduce_Pool_2", bnConvFeatures2, k*4, k*4)
    bnConvFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_Out_BN", bnConvFeatures2, isTraining, False, False)
    poolFeatures2 = mConvBuilder.create_convolution(
        convName="Pool_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=bnConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.8,
        KDEWindow= 0.2)
    
    # Third Convolution
    bnPoolFeatures2 = batch_norm_RELU_drop_out("Conv_3_In_BN", poolFeatures2, isTraining, False, False)
    convFeatures3 = mConvBuilder.create_convolution(
        convName="Conv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=bnPoolFeatures2,
        inNumFeatures=k*4, 
        convRadius=math.sqrt(3))
    convFeatures3 = tf.concat([poolFeatures2, convFeatures3], 1)

    
    ##################################################### Multi-hierarchy sampling
    
    
    # Second upsampling
    bnFeatures3 = batch_norm_RELU_drop_out("Up_2_3_BN", convFeatures3, isTraining, False, False)
    upFeatures2_3 = mConvBuilder.create_convolution(
        convName="Up_2_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=1, 
        inFeatures=bnFeatures3,
        inNumFeatures=k*8, 
        convRadius=0.8)
    deConvFeatures2 = tf.concat([upFeatures2_3, convFeatures2], 1)
    deConvFeatures2 = batch_norm_RELU_drop_out("DeConv_2_Reduce_In_BN", deConvFeatures2, isTraining, False, False)
    deConvFeatures2 = conv_1x1("DeConv_2_Reduce", deConvFeatures2, k*12, k*4)
    deConvFeatures2 = batch_norm_RELU_drop_out("DeConv_2_Reduce_Out_BN", deConvFeatures2, isTraining, False, False)
    deConvFeatures2 = mConvBuilder.create_convolution(
        convName="DeConv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=deConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.4)    
    
    # First upsampling
    bnDeConvFeatures2 = batch_norm_RELU_drop_out("Up_1_2_BN", deConvFeatures2, isTraining, False, False)
    upFeatures1_2 = mConvBuilder.create_convolution(
        convName="Up_1_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=0, 
        inFeatures=bnDeConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.2)
    deConvFeatures1 = tf.concat([upFeatures1_2, convFeatures1], 1)
    deConvFeatures1 = batch_norm_RELU_drop_out("DeConv_1_Reduce_In_BN", deConvFeatures1, isTraining, False, False)
    deConvFeatures1 = conv_1x1("DeConv_1_Reduce", deConvFeatures1, k*5, k*2)
    deConvFeatures1 = batch_norm_RELU_drop_out("DeConv_1_Reduce_Out_BN", deConvFeatures1, isTraining, False, False)
    normals = mConvBuilder.create_convolution(
        convName="DeConv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=deConvFeatures1, 
        inNumFeatures=k*2, 
        outNumFeatures=3,
        convRadius=0.1,
        multiFeatureConv=True)

    return normals
