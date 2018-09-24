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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import MLP_2_hidden, batch_norm_RELU_drop_out, conv_1x1

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, numOutCat, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    ############################################ Compute point hierarchy
    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.1, 0.4, math.sqrt(3.0)+0.1], "MCClass_PH", batchSize)
    

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25)

    # First Convolution
    convFeatures1 = mConvBuilder.create_convolution(
        convName="Conv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k,
        convRadius= 0.1,
        multiFeatureConv=True)

    # First Pooling
    convFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_In_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures1 = conv_1x1("Reduce_Pool_1", convFeatures1, k, k*2)
    convFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_Out_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    poolFeatures1 = mConvBuilder.create_convolution(
        convName="Pool_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        outPointLevel=1, 
        inFeatures=convFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.2,
        KDEWindow= 0.2)

    
    # Second Convolution
    bnPoolFeatures1 = batch_norm_RELU_drop_out("Conv_2_In_BN", poolFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=bnPoolFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.4)
    convFeatures2 = tf.concat([poolFeatures1, convFeatures2], 1)

    # Second Pooling
    convFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_In_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = conv_1x1("Reduce_Pool_2", convFeatures2, k*4, k*8)
    convFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_Out_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    poolFeatures2 = mConvBuilder.create_convolution(
        convName="Pool_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=convFeatures2,
        inNumFeatures=k*8, 
        convRadius=0.8,
        KDEWindow= 0.2)
    

    # Third Convolution
    bnPoolFeatures2 = batch_norm_RELU_drop_out("Conv_3_In_BN", poolFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = mConvBuilder.create_convolution(
        convName="Conv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=bnPoolFeatures2,
        inNumFeatures=k*8, 
        convRadius=1.1)
    convFeatures3 = tf.concat([poolFeatures2, convFeatures3], 1)

    # Third Pooling
    convFeatures3 = batch_norm_RELU_drop_out("Reduce_Pool_3_In_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = conv_1x1("Reduce_Pool_3", convFeatures3, k*16, k*32)
    convFeatures3 = batch_norm_RELU_drop_out("Reduce_Pool_3_Out_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    poolFeatures3 = mConvBuilder.create_convolution(
        convName="Pool_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=3, 
        inFeatures=convFeatures3,
        inNumFeatures=k*32, 
        convRadius=math.sqrt(3.0)+0.1,
        KDEWindow= 0.2)
        
    #Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_final", poolFeatures3, isTraining, useConvDropOut, keepProbConv)
    finalLogits = MLP_2_hidden(finalInput, k*32, k*16, k*8, numOutCat, "Final_Logits", keepProbFull, isTraining, useDropOutFull)

    return finalLogits
