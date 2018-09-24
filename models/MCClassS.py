'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCClassS.py

    \brief Definition of the network architecture MCClassS for classification 
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
    mPointHierarchy = PointHierarchy(points, features, batchIds, [0.1, 0.4, math.sqrt(3.0)+0.1], "MCClassS_PH", batchSize)
    

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.2)

    #### Convolution 1
    convFeatures1 = mConvBuilder.create_convolution(
        convName = "Conv_1", 
        inPointHierarchy = mPointHierarchy,
        inPointLevel=0, 
        outPointLevel=1, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures,
        outNumFeatures=k, 
        convRadius= 0.2,
        multiFeatureConv=True)

    #### Convolution 2
    convFeatures1 = batch_norm_RELU_drop_out("Reduce_1_In_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures1 = conv_1x1("Reduce_1", convFeatures1, k, k*2)
    convFeatures1 = batch_norm_RELU_drop_out("Reduce_1_Out_BN", convFeatures1, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=1, 
        outPointLevel=2, 
        inFeatures=convFeatures1,
        inNumFeatures=k*2, 
        convRadius=0.8)

    #### Convolution 3
    convFeatures2 = batch_norm_RELU_drop_out("Reduce_2_In_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures2 = conv_1x1("Reduce_2", convFeatures2, k*2, k*4)
    convFeatures2 = batch_norm_RELU_drop_out("Reduce_2_Out_BN", convFeatures2, isTraining, useConvDropOut, keepProbConv)
    convFeatures3 = mConvBuilder.create_convolution(
        convName="Conv_3", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=2, 
        outPointLevel=3, 
        inFeatures=convFeatures2,
        inNumFeatures=k*4, 
        convRadius=math.sqrt(3.0)+0.1)

    #Fully connected MLP - Global features.
    finalInput = batch_norm_RELU_drop_out("BNRELUDROP_final", convFeatures3, isTraining, useConvDropOut, keepProbConv)
    finalLogits = MLP_2_hidden(finalInput, k*4, k*2, k, numOutCat, "Final_Logits", keepProbFull, isTraining, useDropOutFull)

    return finalLogits
