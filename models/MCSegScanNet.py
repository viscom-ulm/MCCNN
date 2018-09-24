'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCSeg.py

    \brief Definition of the network architecture MCSegScanNet for  
           segmentation tasks on the ScanNet dataset.

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
from MCNetworkUtils import MLP_2_hidden, batch_norm_RELU_drop_out, conv_1x1

def create_network(points, batchIds, features, numInputFeatures, numSem, batchSize, k, isTraining, 
    keepProbConv, keepProbFull, useConvDropOut = False, useDropOutFull = True):

    ############################################  Compute point hierarchy
    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.1, 0.2, 0.4, 0.8], "MCSegScanNet_PH", batchSize, False)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.25, relativeRadius=False)

    ############################################ Encoder

    # Init pooling
    poolFeatures0 = mConvBuilder.create_convolution(
        convName="Pool_0", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        outPointLevel=1,
        inFeatures=features, 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k,
        convRadius=0.1,
        KDEWindow= 0.2,
        multiFeatureConv=True)  

    # First Convolution
    bnPoolFeatures0 = batch_norm_RELU_drop_out("Conv_1_In_BN", poolFeatures0, isTraining, useConvDropOut, keepProbConv)
    convFeatures1 = mConvBuilder.create_convolution(
        convName="Conv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=bnPoolFeatures0,
        inNumFeatures=k, 
        convRadius=0.4)
    convFeatures1 = tf.concat([poolFeatures0, convFeatures1], 1)
    
    # First Pooling
    bnConvFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_In_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures1 = conv_1x1("Reduce_Pool_1", bnConvFeatures1, k*2, k*2)
    bnConvFeatures1 = batch_norm_RELU_drop_out("Reduce_Pool_1_Out_BN", bnConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    poolFeatures1 = mConvBuilder.create_convolution(
        convName="Pool_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=bnConvFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.4,
        KDEWindow= 0.2)

    # Second Convolution
    bnPoolFeatures1 = batch_norm_RELU_drop_out("Conv_2_In_BN", poolFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=bnPoolFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.8)
    convFeatures2 = tf.concat([poolFeatures1, convFeatures2], 1)
    
    # Second Pooling
    bnConvFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_In_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures2 = conv_1x1("Reduce_Pool_2", bnConvFeatures2, k*4, k*4)
    bnConvFeatures2 = batch_norm_RELU_drop_out("Reduce_Pool_2_Out_BN", bnConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    poolFeatures2 = mConvBuilder.create_convolution(
        convName="Pool_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=3, 
        inFeatures=bnConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.8,
        KDEWindow= 0.2)
    
    # Third Convolution
    bnPoolFeatures2 = batch_norm_RELU_drop_out("Conv_3_In_BN", poolFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = mConvBuilder.create_convolution(
        convName="Conv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3, 
        inFeatures=bnPoolFeatures2,
        inNumFeatures=k*4, 
        convRadius=1.6)
    convFeatures3 = tf.concat([poolFeatures2, convFeatures3], 1)

    # Third Pooling
    bnConvFeatures3 = batch_norm_RELU_drop_out("Reduce_Pool_3_In_BN", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures3 = conv_1x1("Reduce_Pool_3", bnConvFeatures3, k*8, k*8)
    bnConvFeatures3 = batch_norm_RELU_drop_out("Reduce_Pool_3_Out_BN", bnConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    poolFeatures3 = mConvBuilder.create_convolution(
        convName="Pool_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3, 
        outPointLevel=4, 
        inFeatures=bnConvFeatures3,
        inNumFeatures=k*8, 
        convRadius=1.6,
        KDEWindow= 0.2)
    
    # Fourth Convolution
    bnPoolFeatures3 = batch_norm_RELU_drop_out("Conv_4_In_BN", poolFeatures3, isTraining, useConvDropOut, keepProbConv)
    convFeatures4 = mConvBuilder.create_convolution(
        convName="Conv_4", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=4, 
        inFeatures=bnPoolFeatures3,
        inNumFeatures=k*8, 
        convRadius=5.0)
    convFeatures4 = tf.concat([poolFeatures3, convFeatures4], 1)

    
    ############################################ Decoder
    
    # Third upsampling
    bnConvFeatures4 = batch_norm_RELU_drop_out("Up3_4_Reduce_In_BN", convFeatures4, isTraining, useConvDropOut, keepProbConv)
    bnConvFeatures4 = conv_1x1("Up3_4_Reduce", bnConvFeatures4, k*16, k*8)
    bnConvFeatures4 = batch_norm_RELU_drop_out("Up3_4_Reduce_Out_BN", bnConvFeatures4, isTraining, useConvDropOut, keepProbConv)
    upFeatures3_4 = mConvBuilder.create_convolution(
        convName="Up_3_4", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=4,
        outPointLevel=3, 
        inFeatures=bnConvFeatures4,
        inNumFeatures=k*8, 
        convRadius=1.6)
    upFeatures3_4 = tf.concat([upFeatures3_4, convFeatures3], 1)
    deConvFeatures3 = batch_norm_RELU_drop_out("DeConv_3_Reduce_In_BN", upFeatures3_4, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures3 = conv_1x1("DeConv_3_Reduce", deConvFeatures3, k*16, k*8)
    deConvFeatures3 = batch_norm_RELU_drop_out("DeConv_3_Reduce_Out_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures3 = mConvBuilder.create_convolution(
        convName="DeConv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3, 
        inFeatures=deConvFeatures3,
        inNumFeatures=k*8, 
        convRadius=1.6)   

    
    # Second upsampling
    bnDeConvFeatures3 = batch_norm_RELU_drop_out("Up2_3_Reduce_In_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    bnDeConvFeatures3 = conv_1x1("Up2_3_Reduce", bnDeConvFeatures3, k*8, k*4)
    bnDeConvFeatures3 = batch_norm_RELU_drop_out("Up2_3_Reduce_Out_BN", bnDeConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    upFeatures2_3 = mConvBuilder.create_convolution(
        convName="Up_2_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3,
        outPointLevel=2, 
        inFeatures=bnDeConvFeatures3,
        inNumFeatures=k*4, 
        convRadius=0.8)
    upFeatures2_3 = tf.concat([upFeatures2_3, convFeatures2], 1)
    deConvFeatures2 = batch_norm_RELU_drop_out("DeConv_2_Reduce_In_BN", upFeatures2_3, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures2 = conv_1x1("DeConv_2_Reduce", deConvFeatures2, k*8, k*4)
    deConvFeatures2 = batch_norm_RELU_drop_out("DeConv_2_Reduce_Out_BN", deConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures2 = mConvBuilder.create_convolution(
        convName="DeConv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        inFeatures=deConvFeatures2,
        inNumFeatures=k*4, 
        convRadius=0.8)
    
    
    # First multiple upsamplings
    bnDeConvFeatures2 = batch_norm_RELU_drop_out("Up1_2_Reduce_In_BN", deConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    bnDeConvFeatures2 = conv_1x1("Up1_2_Reduce", bnDeConvFeatures2, k*4, k*2)
    bnDeConvFeatures2 = batch_norm_RELU_drop_out("Up1_2_Reduce_Out_BN", bnDeConvFeatures2, isTraining, useConvDropOut, keepProbConv)
    upFeatures1_2 = mConvBuilder.create_convolution(
        convName="Up_1_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2,
        outPointLevel=1, 
        inFeatures=bnDeConvFeatures2,
        inNumFeatures=k*2, 
        convRadius=0.4)
    bnDeConvFeatures3 = batch_norm_RELU_drop_out("Up1_3_Reduce_In_BN", deConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    bnDeConvFeatures3 = conv_1x1("Up1_3_Reduce", bnDeConvFeatures3, k*8, k*2)
    bnDeConvFeatures3 = batch_norm_RELU_drop_out("Up1_3_Reduce_Out_BN", bnDeConvFeatures3, isTraining, useConvDropOut, keepProbConv)
    upFeatures1_3 = mConvBuilder.create_convolution(
        convName="Up_1_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=3,
        outPointLevel=1, 
        inFeatures=bnDeConvFeatures3,
        inNumFeatures=k*2, 
        convRadius=0.8)
    bnDeConvFeatures4 = batch_norm_RELU_drop_out("Up1_4_Reduce_In_BN", convFeatures4, isTraining, useConvDropOut, keepProbConv)
    bnDeConvFeatures4 = conv_1x1("Up1_4_Reduce", bnDeConvFeatures4, k*16, k*2)
    bnDeConvFeatures4 = batch_norm_RELU_drop_out("Up1_4_Reduce_Out_BN", bnDeConvFeatures4, isTraining, useConvDropOut, keepProbConv)
    upFeatures1_4 = mConvBuilder.create_convolution(
        convName="Up_1_4", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=4,
        outPointLevel=1, 
        inFeatures=bnDeConvFeatures4,
        inNumFeatures=k*2, 
        convRadius=1.6)
    upFeatures1 = tf.concat([upFeatures1_4, upFeatures1_3, upFeatures1_2, convFeatures1], 1)
    deConvFeatures1 = batch_norm_RELU_drop_out("DeConv_1_Reduce_In_BN", upFeatures1, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures1 = conv_1x1("DeConv_1_Reduce", deConvFeatures1, k*8, k*4)
    deConvFeatures1 = batch_norm_RELU_drop_out("DeConv_1_Reduce_Out_BN", deConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    deConvFeatures1 = mConvBuilder.create_convolution(
        convName="DeConv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        inFeatures=deConvFeatures1,
        inNumFeatures=k*4, 
        convRadius=0.4)
    deConvFeatures1 = tf.concat([upFeatures1_4, upFeatures1_3, upFeatures1_2, convFeatures1, deConvFeatures1], 1)

    
    # Final upsampling
    upFeaturesFinal = batch_norm_RELU_drop_out("Up_Final_Reduce_In_BN", deConvFeatures1, isTraining, useConvDropOut, keepProbConv)
    upFeaturesFinal = conv_1x1("Up_Final_Reduce", upFeaturesFinal, k*12, k*4)
    upFeaturesFinal = batch_norm_RELU_drop_out("Up_Final_Reduce_Out_BN", upFeaturesFinal, isTraining, useConvDropOut, keepProbConv)
    finalFeatures = mConvBuilder.create_convolution(
        convName="Up_0_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1,
        outPointLevel=0, 
        inFeatures=upFeaturesFinal,
        inNumFeatures=k*4, 
        convRadius=0.1)

    
    # Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_hier_final", finalFeatures, isTraining, useConvDropOut, keepProbConv)
    finalLogits = MLP_2_hidden(finalInput, k*4, k*4, k*2, numSem, "Final_Logits", keepProbFull, isTraining, useDropOutFull, useInitBN = False)

    return finalLogits
