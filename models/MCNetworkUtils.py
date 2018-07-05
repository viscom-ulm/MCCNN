'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file MCNetworkUtils.py

    \brief Different util functions used to define network architectures.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import tensorflow as tf
import os
import sys
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))
from MCConvModule import sort_points_step1, sort_points_step2, find_neighbors, \
    compute_pdf, poisson_sampling, get_sampled_features, spatial_conv, \
    get_block_size, transform_indexs

############################################################################# Utils

def fully_connected(features, numInputFeatures, hidden1_units, hidden2_units, outClasses, layerName, keepProb, isTraining, useDropOut = False, useInitBN = True):
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)

    if useInitBN:
        features = tf.layers.batch_normalization(inputs = features, training = isTraining, name = layerName+"_BN_Init")
    
    # Hidden 1
    with tf.name_scope(layerName+'_hidden1'):
        weights = tf.Variable(initializer([numInputFeatures, hidden1_units]), name='weights')
        tf.add_to_collection('weight_decay_loss', weights)
        biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
        mul1 = tf.matmul(features, weights) + biases
        mul1 = tf.layers.batch_normalization(inputs = mul1, training = isTraining, name = layerName+"_BN_h1")
        hidden1 = tf.nn.relu(mul1)
        
    # Hidden 2
    with tf.name_scope(layerName+'_hidden2'):
        if useDropOut:
            hidden1 = tf.nn.dropout(hidden1, keepProb)
        weights = tf.Variable(initializer([hidden1_units, hidden2_units]), name='weights')
        tf.add_to_collection('weight_decay_loss', weights)
        biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
        mul2 = tf.matmul(hidden1, weights) + biases
        mul2 = tf.layers.batch_normalization(inputs = mul2, training = isTraining, name = layerName+"_BN_h2")
        hidden2 = tf.nn.relu(mul2)
        
    # Linear
    with tf.name_scope(layerName+'_softmax_linear'):
        if useDropOut:
            hidden2 = tf.nn.dropout(hidden2, keepProb)
        weights = tf.Variable(initializer([hidden2_units, outClasses]), name='weights')
        tf.add_to_collection('weight_decay_loss', weights)
        biases = tf.Variable(tf.zeros([outClasses]), name='biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def reduce_channels(layerName, inputs, numInputs, numOutFeatures):

    with tf.name_scope(layerName+'_reduction'):
        initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)
        weights = tf.Variable(initializer([numInputs, numOutFeatures]), name='weights')
        tf.add_to_collection('weight_decay_loss', weights)
        biases = tf.Variable(tf.zeros([numOutFeatures]), name='biases')
        reducedOutput = tf.matmul(inputs, weights) + biases
    return reducedOutput

def batch_norm_RELU_drop_out(layerName, inFeatures, isTraining, usedDropOut, keepProb):
    with tf.name_scope(layerName):
        inFeatures = tf.layers.batch_normalization(inputs = inFeatures, training = isTraining, name = layerName+"_BN")
        inFeatures = tf.nn.relu(inFeatures)
        if usedDropOut:
            inFeatures = tf.nn.dropout(inFeatures, keepProb)
        return inFeatures

############################################################################# Grid pt distribution

def distribute_grid(layerName, inPts, inBatchs, inFeatures, aabbMin, aabbMax, batchSize, radius):
    numCells = max(int(math.floor(1.0/radius)), 1)
    with tf.name_scope(layerName+'_grid'):
        keys, indexs = sort_points_step1(inPts, inBatchs, aabbMin, aabbMax, numCells, batchSize)
        sortPts, sortBatchs, sortFeatures, cellIndexs = sort_points_step2(inPts, inBatchs, inFeatures, keys, indexs, numCells, batchSize)
        return sortPts, sortBatchs, sortFeatures, cellIndexs, indexs

############################################################################# Monte Carlo convolutions

def prepare_conv(layerName, sampledPts, sampledBatchIds, sortPts, sortBatchIds, cellIndexs, aabbMin, aabbMax, batchSize, radius, pdfKernelWindow):
    with tf.name_scope(layerName):
        startIndexs, packedNeighs = find_neighbors(sampledPts, sampledBatchIds, sortPts, cellIndexs, aabbMin, aabbMax, radius, batchSize)
        pdfs = compute_pdf(sortPts, sortBatchIds, aabbMin, aabbMax, startIndexs, packedNeighs, pdfKernelWindow, radius, batchSize)
        return pdfs, startIndexs, packedNeighs

def compute_convolution(convName, sortPts, sortFeatures, sortBatchIds, inPDfs, inSampledPts, startIndexNeigh, packedNeighs, aabbMin, aabbMax, batchSize, radius, 
    numFeatures, numOutFeatures, isTraining, fullConv = True):
    
    initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)

    blockSize = get_block_size()

    if fullConv:
            numOutNeurons = numFeatures*numOutFeatures
    else:
        numOutNeurons = numFeatures
    numBlocks = int(numOutNeurons/blockSize)
    if numOutNeurons%blockSize!=0:
        numBlocks = numBlocks+1

    weights = tf.get_variable(convName+'_weights', [3, blockSize*numBlocks], initializer=initializer)
    tf.add_to_collection('weight_decay_loss', weights)
    biases = tf.get_variable(convName+'_biases', [blockSize*numBlocks], initializer=initializer)
    weights2 = tf.get_variable(convName+'_weights2', [numBlocks, blockSize, blockSize], initializer=initializer)
    weights2 = tf.reshape(weights2, [blockSize, numBlocks*blockSize])
    tf.add_to_collection('weight_decay_loss', weights2)
    biases2 = tf.get_variable(convName+'_biases2', [numBlocks, blockSize], initializer=initializer)
    biases2 = tf.reshape(biases2, [numBlocks*blockSize])
    weights3 = tf.get_variable(convName+'_weights3', [numBlocks, blockSize, blockSize], initializer=initializer)
    weights3 = tf.reshape(weights3, [blockSize, numBlocks*blockSize])
    tf.add_to_collection('weight_decay_loss', weights3)
    biases3 = tf.get_variable(convName+'_biases3', [numBlocks, blockSize], initializer=initializer)
    biases3 = tf.reshape(biases3, [numBlocks*blockSize])

    return spatial_conv(sortPts, sortFeatures, sortBatchIds, inPDfs, inSampledPts, startIndexNeigh, packedNeighs, aabbMin, aabbMax,
        weights, weights2, weights3, biases, biases2, biases3, numOutFeatures, fullConv, batchSize, radius)

############################################################################# Pooling

def sampling_and_conv_pooling(layerName, sortPts, sortFeatures, sortBatchs, cellIndexs, aabbMin, aabbMax, radius, radiusPool, numFeatures,
    isTraining, useDropOut, keepProb, batchSize, pdfKernelWindow):
    with tf.name_scope(layerName):
        auxSortPts = sortPts
        auxSortFeatures = sortFeatures
        auxSortBatchIds = sortBatchs
        auxCellIndexs = cellIndexs
        sampledPts, sampledBatchsIds, _ = poisson_sampling(auxSortPts, auxSortBatchIds, auxCellIndexs, aabbMin, aabbMax, radius, batchSize)
        if radiusPool != radius:
            auxSortPts, auxSortBatchIds, auxSortFeatures, auxCellIndexs, _ = distribute_grid(layerName+"_sroting", auxSortPts, auxSortBatchIds, auxSortFeatures,
                aabbMin, aabbMax, batchSize, radiusPool)
        poolStartIndexs, poolPackedNeighs = find_neighbors(sampledPts, sampledBatchsIds, auxSortPts, auxCellIndexs, aabbMin, aabbMax, radius, batchSize)
        poolPdfs = compute_pdf(auxSortPts, auxSortBatchIds, aabbMin, aabbMax, poolStartIndexs, poolPackedNeighs, pdfKernelWindow, radius, batchSize)
        resConvFeatures = batch_norm_RELU_drop_out(layerName+"_BNRELUDROP", auxSortFeatures, isTraining, useDropOut, keepProb)
        sampledFeatures = compute_convolution(layerName+"_Pool_Conv", auxSortPts, resConvFeatures, auxSortBatchIds, poolPdfs, sampledPts, poolStartIndexs, poolPackedNeighs,
            aabbMin, aabbMax, batchSize, radiusPool, numFeatures, numFeatures, isTraining, fullConv = False)
        return sampledPts, sampledBatchsIds, sampledFeatures

def sampling_center_and_conv_pooling(layerName, pts, features, batchs, aabbMin, aabbMax, numFeatures, isTraining, useDropOut, keepProb, batchSize, pdfKernelWindow):
    with tf.name_scope(layerName):
        sortPts, sortBatchs, sortFeatures, cellIndexs, _ = distribute_grid(layerName+'_Grid', pts, batchs, features, aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1)
        sampledPts, sampledBatchsIds, _ = poisson_sampling(sortPts, sortBatchs, cellIndexs, aabbMin, aabbMax, math.sqrt(3.0)+0.1, batchSize)
        centerPts = (aabbMax + aabbMin)*0.5
        centerPts = tf.slice(centerPts, [0, 0], tf.shape(sampledPts))
        poolStartIndexs, poolPackedNeighs = find_neighbors(centerPts, sampledBatchsIds, sortPts, cellIndexs, aabbMin, aabbMax, math.sqrt(3.0)+0.1, batchSize)
        poolPdfs = compute_pdf(sortPts, sortBatchs, aabbMin, aabbMax, poolStartIndexs, poolPackedNeighs, pdfKernelWindow, math.sqrt(3.0)+0.1, batchSize)
        resConvFeatures = batch_norm_RELU_drop_out(layerName+"_BNRELUDROP", sortFeatures, isTraining, useDropOut, keepProb)
        sampledFeatures = compute_convolution(layerName+"_Pool_Conv", sortPts, resConvFeatures, sortBatchs, poolPdfs, centerPts, poolStartIndexs, poolPackedNeighs,
            aabbMin, aabbMax, batchSize, math.sqrt(3.0)+0.1, numFeatures, numFeatures, isTraining, fullConv = False)
        return centerPts, sampledBatchsIds, sampledFeatures


############################################################################# Computing hierarchy

def compute_hierarchy(layerName, inPts, inBatchIds, inFeatures, radiusList, aabbMin, aabbMax, batchSize):
    with tf.name_scope(layerName):
        
        outSampledPts = [inPts]
        outSampledBatchIds = [inBatchIds]
        outSampledFeatures = [inFeatures]
        outSampledIndexs = []

        currPts = inPts
        currBatchIds = inBatchIds
        currFeatures = inFeatures

        for currRadius in radiusList:
            sortPts, sortBatchs, sortFeatures, cellIndexs, indexs = distribute_grid(
                layerName+"_"+str(currRadius), currPts, currBatchIds, currFeatures, aabbMin, aabbMax, batchSize, currRadius)

            sampledPts, sampledBatchsIds, sampledIndexs = poisson_sampling(
                sortPts, sortBatchs, cellIndexs, aabbMin, aabbMax, currRadius, batchSize)
            sampledFeatures = get_sampled_features(sampledIndexs, sortFeatures)
            transformedIndexs = transform_indexs(sampledIndexs, indexs)

            outSampledPts.append(sampledPts)
            outSampledBatchIds.append(sampledBatchsIds)
            outSampledFeatures.append(sampledFeatures)
            outSampledIndexs.append(transformedIndexs)

            currPts = sampledPts
            currBatchIds = sampledBatchsIds
            currFeatures = sampledFeatures

    return outSampledPts, outSampledBatchIds, outSampledFeatures, outSampledIndexs

def compute_hierarchy_features(layerName, inFeatures, inIndexs):
    with tf.name_scope(layerName):

        outSampledFeatures = [inFeatures]
        currFeatures = inFeatures

        for currIndex  in inIndexs:
            currFeatures = get_sampled_features(currIndex, currFeatures)
            outSampledFeatures.append(currFeatures)

        return outSampledFeatures