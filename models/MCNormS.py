'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCNormS.py

    \brief Definition of the network architecture MCNormS for normal
            estimation tasks.

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
from MCNetworkUtils import batch_norm_RELU_drop_out

def create_network(points, batchIds, features, numInputFeatures, batchSize, k, isTraining, multiConv = True, useMC = True):

    ############################################  Compute point hierarchy
    mPointHierarchy = PointHierarchy(points, features, batchIds, [], "MCNormS_PH", batchSize)

    ############################################ Convolutions
    mConvBuilder = ConvolutionBuilder(KDEWindow=0.2)

    # Convolution 1
    convFeatures1 = mConvBuilder.create_convolution(
        convName="Conv_1", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=features, 
        inNumFeatures=numInputFeatures, 
        outNumFeatures=k,
        convRadius=0.15,
        multiFeatureConv=True)

    #BatchNorm and RELU
    convFeatures1 = batch_norm_RELU_drop_out("BN_RELU", convFeatures1, isTraining, False, False)
    
    # Convolution 2
    normals = mConvBuilder.create_convolution(
        convName="Conv_2", 
        inPointHierarchy=mPointHierarchy,
        inPointLevel=0, 
        inFeatures=convFeatures1, 
        inNumFeatures=k, 
        outNumFeatures=3,
        convRadius=0.15,
        multiFeatureConv=True)

    return normals
