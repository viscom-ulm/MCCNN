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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from MCConvBuilder import PointHierarchy, ConvolutionBuilder
from MCNetworkUtils import MLP_2_hidden, batch_norm_RELU_drop_out, conv_1x1

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, numOutCat, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    ############################################ Compute point hierarchy
    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.1, 0.4, math.sqrt(3.0)+0.1], "MCClassH_PH", batchSize)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25)

    ############################################ LOGITS 1

    ############################################ First level 

    # Convolution
    convFeatures1 = mConvBuilder.create_convolution(
        convName="Conv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k,
        convRadius= 0.1,
        multiFeatureConv=True)

    # Pooling
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

    ############################################ Second level convolutions

    #### Convolution
    bnPoolFeatures1 = batch_norm_RELU_drop_out("Reduce_Conv_2_In_BN", poolFeatures1, isTraining, useConvDropOut, keepProbConv)
    bnPoolFeatures1 = conv_1x1("Reduce_Conv_2", bnPoolFeatures1, k*2, k*2)
    bnPoolFeatures1 = batch_norm_RELU_drop_out("Reduce_Conv_2_Out_BN", bnPoolFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=bnPoolFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.4)
    convFeatures2 = tf.concat([poolFeatures1, convFeatures2], 1)

    # Pooling
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

    
    ############################################ Third level convolutions

    # Convolution
    bnPoolFeatures2 = batch_norm_RELU_drop_out("Reduce_Conv_3_In_BN", poolFeatures2, isTraining, useConvDropOut, keepProbConv)
    bnPoolFeatures2 = conv_1x1("Reduce_Conv_3", bnPoolFeatures2, k*8, k*8)
    bnPoolFeatures2 = batch_norm_RELU_drop_out("Reduce_Conv_3_Out_BN", bnPoolFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = mConvBuilder.create_convolution(
        convName="Conv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=bnPoolFeatures2,
        inNumFeatures=k*8, 
        convRadius=1.2)
    convFeatures3 = tf.concat([poolFeatures2, convFeatures3], 1)

    # Pooling
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
    finalLogits1 = MLP_2_hidden(finalInput, k*32, k*16, k*8, numOutCat, "Final_Logits", keepProbFull, isTraining, useDropOutFull)


    ############################################ LOGITS 2

    ############################################ Second level convolutions

    #### Convolution
    convFeatures22 = mConvBuilder.create_convolution(
        convName="Conv_2_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=mPointHierarchy.features_[1], 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k*2,
        convRadius= 0.4,
        multiFeatureConv=True)

    # Pooling
    convFeatures22 = batch_norm_RELU_drop_out("Reduce_Pool_2_2_In_BN", convFeatures22, isTraining, useConvDropOut, keepProbConv)
    convFeatures22 = conv_1x1("Reduce_Pool_2_2", convFeatures22, k*2, k*8)
    convFeatures22 = batch_norm_RELU_drop_out("Reduce_Pool_2_2_Out_BN", convFeatures22, isTraining, useConvDropOut, keepProbConv)
    poolFeatures22 = mConvBuilder.create_convolution(
        convName="Pool_2_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=convFeatures22,
        inNumFeatures=k*8, 
        convRadius=0.8,
        KDEWindow= 0.2)

    
    ############################################ Third level convolutions

    # Convolution
    bnPoolFeatures22 = batch_norm_RELU_drop_out("Reduce_Conv_3_2_In_BN", poolFeatures22, isTraining, useConvDropOut, keepProbConv)
    bnPoolFeatures22 = conv_1x1("Reduce_Conv_3_2", bnPoolFeatures22, k*8, k*8)
    bnPoolFeatures22 = batch_norm_RELU_drop_out("Reduce_Conv_3_2_Out_BN", bnPoolFeatures22, isTraining, useConvDropOut, keepProbConv)
    convFeatures32 = mConvBuilder.create_convolution(
        convName="Conv_3_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=bnPoolFeatures22,
        inNumFeatures=k*8, 
        convRadius=1.2)
    convFeatures32 = tf.concat([poolFeatures22, convFeatures32], 1)

    # Pooling
    convFeatures32 = batch_norm_RELU_drop_out("Reduce_Pool_3_2_In_BN", convFeatures32, isTraining, useConvDropOut, keepProbConv)
    convFeatures32 = conv_1x1("Reduce_Pool_3_2", convFeatures32, k*16, k*32)
    convFeatures32 = batch_norm_RELU_drop_out("Reduce_Pool_3_2_Out_BN", convFeatures32, isTraining, useConvDropOut, keepProbConv)
    poolFeatures32 = mConvBuilder.create_convolution(
        convName="Pool_3_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=3, 
        inFeatures=convFeatures32,
        inNumFeatures=k*32, 
        convRadius=math.sqrt(3.0)+0.1,
        KDEWindow= 0.2)

    #Fully connected MLP - Global features.
    finalInput2 = batch_norm_RELU_drop_out("2BNRELUDROP_final", poolFeatures32, isTraining, useConvDropOut, keepProbConv)
    finalLogits2 = MLP_2_hidden(finalInput2, k*32, k*16, k*8, numOutCat, "2Final_Logits", keepProbFull, isTraining, useDropOutFull)

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
