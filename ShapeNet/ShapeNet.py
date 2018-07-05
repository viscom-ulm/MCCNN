'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ShapeNet.py

    \brief Code to train a segmentation network on the ShapeNet dataset.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import time
import argparse
import copy
import random 
import importlib
import os
import json
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

current_milli_time = lambda: time.time() * 1000.0


def create_loss(logits, labels, weigthDecay):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    xentropyloss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    regularizer = tf.contrib.layers.l2_regularizer(scale=weigthDecay)
    regVariables = tf.get_collection('weight_decay_loss')
    regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
    return xentropyloss, regTerm


def create_accuracy(logits, labels, scope):
    _, logitsIndexs = tf.nn.top_k(logits)
    with tf.variable_scope(scope):
        return tf.metrics.accuracy(tf.reshape(labels, [-1, 1]), logitsIndexs)


def create_trainning(lossGraph, learningRate, maxLearningRate, learningDecayFactor, learningRateDecay, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningRateDecay, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph, global_step=global_step)
    return train_op, learningRateExp


def load_model(modelsPath, catId, batchId, outPts, outBatchIds, outFeatures, outCatLabels, outLabels, keepPointProb, augmentData, training):
    with open(modelsPath+".txt", 'r') as modelFile:
        maxPt = np.array([-10000.0, -10000.0, -10000.0])
        minPt = np.array([10000.0, 10000.0, 10000.0])
        pts = []
        for line in modelFile:
            rndNum = np.random.random()
            if rndNum < keepPointProb:
                line = line.replace("\n", "")
                currPoint = line.split()
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                maxPt = np.maximum(auxPoint, maxPt)
                minPt = np.minimum(auxPoint, minPt)
                pts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                outFeatures.append([1])
                outLabels.append(int(float(currPoint[6])))
                outCatLabels.append([catId])
                outBatchIds.append([batchId])

        modelSize = np.subtract(maxPt, minPt)
        for pt in pts:
            if training and augmentData:
                displ = np.multiply(np.random.randn(3), modelSize*0.005)
            else:
                displ = [0.0, 0.0, 0.0]
            outPts.append([pt[0]+displ[0], pt[1]+displ[1], pt[2]+displ[2]])
            
def get_train_and_test_files():
    #Determine the number of categories.
    cat = []
    with open("./shape_data/synsetoffset2category.txt", 'r') as nameFile:
        for line in nameFile:
            strings = line.replace("\n", "").split("\t")
            cat.append((strings[0], strings[1]))
    print(cat)
    segClasses = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 
        'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 
        'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

    #List of files
    trainFiles = []
    with open("./shape_data/train_test_split/shuffled_train_file_list.json", 'r') as f:
        trainFiles = list([d for d in json.load(f)])
    valFiles = []
    with open("./shape_data/train_test_split/shuffled_val_file_list.json", 'r') as f:
        valFiles = list([d for d in json.load(f)])
    testFiles = []
    with open("./shape_data/train_test_split/shuffled_test_file_list.json", 'r') as f:
        testFiles = list([d for d in json.load(f)])
    trainFiles = trainFiles + valFiles

    return cat, segClasses, trainFiles, testFiles

def visualize_progress(val, maxVal, description="", barWidth=20):

    progress = int((val*barWidth) / maxVal)
    progressBar = ['='] * (progress) + ['>'] + ['.'] * (barWidth - (progress+1))
    progressBar = ''.join(progressBar)
    initBar = "%4d/%4d" % (val + 1, maxVal)
    print(initBar + ' [' + progressBar + '] ' + description)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train MCCNN for segmentation tasks (ShapeNet)')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--model', default='MCSeg', help='model (default: MCSeg)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--batchSize', default=32, type=int, help='Batch size  (default: 32)')
    parser.add_argument('--maxEpoch', default=200, type=int, help='Max Epoch  (default: 200)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDeacyFactor', default=0.2, type=float, help='Learning deacy factor (default: 0.2)')
    parser.add_argument('--learningDecayRate', default=25, type=int, help='Learning decay rate  (default: 25 Epochs)')
    parser.add_argument('--maxLearningRate', default=0.00001, type=float, help='Maximum Learning rate (default: 0.00001)')
    parser.add_argument('--useDropOut', action='store_true', help='Use drop out  (default: True)')
    parser.add_argument('--dropOutKeepProb', default=0.5, type=float, help='Keep neuron probabillity drop out  (default: 0.5)')
    parser.add_argument('--useDropOutConv', action='store_true', help='Use drop out in convolution layers (default: False)')
    parser.add_argument('--dropOutKeepProbConv', default=0.8, type=float, help='Keep neuron probabillity drop out in convolution layers (default: 0.8)')
    parser.add_argument('--weightDecay', default=0.0, type=float, help='Weight decay (default: 0.0)')
    parser.add_argument('--ptDropOut', default=0.8, type=float, help='Point drop out (default: 0.8)')
    parser.add_argument('--augment', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    args = parser.parse_args()

    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp ../models/%s.py %s' % (args.model, args.logFolder))
    os.system('cp ShapeNet.py %s' % (args.logFolder))
    logFile = args.logFolder+"/log.txt"

    #Write execution info.
    with open(logFile, "a") as myFile:
        myFile.write("Model: "+args.model+"\n")
        myFile.write("Grow: "+str(args.grow)+"\n")
        myFile.write("BatchSize: "+str(args.batchSize)+"\n")
        myFile.write("MaxEpoch: "+str(args.maxEpoch)+"\n")
        myFile.write("InitLearningRate: "+str(args.initLearningRate)+"\n")
        myFile.write("LearningDeacyFactor: "+str(args.learningDeacyFactor)+"\n")
        myFile.write("LearningDecayRate: "+str(args.learningDecayRate)+"\n")
        myFile.write("MaxLearningRate: "+str(args.maxLearningRate)+"\n")
        myFile.write("UseDropOut: "+str(args.useDropOut)+"\n")
        myFile.write("DropOutKeepProb: "+str(args.dropOutKeepProb)+"\n")
        myFile.write("UseDropOutConv: "+str(args.useDropOutConv)+"\n")
        myFile.write("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv)+"\n")
        myFile.write("WeightDecay: "+str(args.weightDecay)+"\n")
        myFile.write("ptDropOut: "+str(args.ptDropOut)+"\n")
        myFile.write("Augment: "+str(args.augment)+"\n")

    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("BatchSize: "+str(args.batchSize))
    print("MaxEpoch: "+str(args.maxEpoch))
    print("InitLearningRate: "+str(args.initLearningRate))
    print("LearningDeacyFactor: "+str(args.learningDeacyFactor))
    print("LearningDecayRate: "+str(args.learningDecayRate))
    print("MaxLearningRate: "+str(args.maxLearningRate))
    print("UseDropOut: "+str(args.useDropOut))
    print("DropOutKeepProb: "+str(args.dropOutKeepProb))
    print("UseDropOutConv: "+str(args.useDropOutConv))
    print("DropOutKeepProbConv: "+str(args.dropOutKeepProbConv))
    print("WeightDecay: "+str(args.weightDecay))
    print("ptDropOut: "+str(args.ptDropOut))
    print("Augment: "+str(args.augment))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    cat, segClasses, trainFiles, testFiles = get_train_and_test_files()
    numTrainModels = len(trainFiles)
    numBatchesXEpoch = numTrainModels/args.batchSize
    if numTrainModels%args.batchSize != 0:
        numBatchesXEpoch = numBatchesXEpoch + 1
    numTestModels = len(testFiles)
    print("Train models: " + str(numTrainModels))
    print("Test models: " + str(numTestModels))

    #Create variable and place holders
    global_step = tf.Variable(0, name='global_step', trainable=False)
    inPts = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None, 1])
    inFeatures = tf.placeholder(tf.float32, [None, 1])
    inCatLabels = tf.placeholder(tf.int32, [None, 1])
    inLabels = tf.placeholder(tf.int32, [None])
    isTraining = tf.placeholder(tf.bool)
    keepProbConv = tf.placeholder(tf.float32)
    keepProbFull = tf.placeholder(tf.float32)
    iouVal = tf.placeholder(tf.float32)

    #Create the network
    logits = model.create_network(inPts, inBatchIds, inFeatures, inCatLabels, 1, len(cat), 50, args.batchSize, 
        args.grow, isTraining, keepProbConv, keepProbFull, args.useDropOutConv, args.useDropOut)
          
    #Create predict labels
    predictedLabels = tf.argmax(logits, 1)
    
    #Create loss
    xentropyLoss, regularizationLoss = create_loss(logits, inLabels, args.weightDecay)
    loss = xentropyLoss + regularizationLoss

    #Create training
    trainning, learningRateExp = create_trainning(loss, 
        args.initLearningRate, args.maxLearningRate, args.learningDeacyFactor, 
        args.learningDecayRate*numBatchesXEpoch, global_step)
    learningRateSumm = tf.summary.scalar('learninRate', learningRateExp)

    #Create accuracy metric
    accuracyVal, accuracyAccumOp = create_accuracy(logits, inLabels, 'metrics')
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

    #Create sumaries
    lossSummary = tf.summary.scalar('loss', loss)
    xEntropyLossSummary = tf.summary.scalar('loss_XEntropy', xentropyLoss)
    regularizationLossSummary = tf.summary.scalar('loss_Regularization', regularizationLoss)
    trainingSummary = tf.summary.merge([lossSummary, xEntropyLossSummary, regularizationLossSummary, learningRateSumm])
    metricsSummary = tf.summary.scalar('accuracy', accuracyVal)
    metricsTestSummary = tf.summary.merge([tf.summary.scalar('Tes_Accuracy', accuracyVal), tf.summary.scalar('Test_IoU', iouVal)], name='TestMetrics')

    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver()
    
    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #Create the summary writer
    summary_writer = tf.summary.FileWriter(args.logFolder, sess.graph)
    summary_writer.add_graph(sess.graph)
    
    #Init variables
    sess.run(init)
    sess.run(initLocal)
    step = 0
    epochStep = 0
    np.random.seed(int(time.time()))
    
    #Train
    for epoch in range(args.maxEpoch):
        cpyTrainFiles = copy.copy(trainFiles)

        startEpochTime = current_milli_time()
        startTrainTime = current_milli_time()

        epochStep = 0
        lossInfoCounter = 0
        lossAccumValue = 0.0

        sess.run(resetMetrics)

        #Iterate over all the train files
        while len(cpyTrainFiles) > 0:
    
            points = []
            batchIds = []
            features = []
            catLabels = []
            labels = []
            for i in range(args.batchSize):
                if len(cpyTrainFiles) > 0:
                    currentModelIndex = random.randint(0, len(cpyTrainFiles)-1)

                    currentModel = cpyTrainFiles[currentModelIndex]
                    cpyTrainFiles.pop(currentModelIndex)
                    
                    catId = 0
                    for currCat in range(len(cat)):
                        if cat[currCat][1] in currentModel:
                            catId = currCat
                    
                    load_model(currentModel, catId, i, points, batchIds, features, catLabels, labels, args.ptDropOut, args.augment, True)
                    
            _, lossRes, xentropyLossRes, regularizationLossRes, trainingSummRes, _ = \
                sess.run([trainning, loss, xentropyLoss, regularizationLoss, trainingSummary, accuracyAccumOp], 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels, inLabels: labels, 
                isTraining: True, keepProbConv: args.dropOutKeepProbConv, keepProbFull: args.dropOutKeepProb})


            summary_writer.add_summary(trainingSummRes, step)

            lossAccumValue += lossRes
            lossInfoCounter += 1

            if lossInfoCounter == 10:
                endTrainTime = current_milli_time()                   
                currAccuracy, metricsSummRes = sess.run([accuracyVal, metricsSummary])
                summary_writer.add_summary(metricsSummRes, step)

                visualize_progress(epochStep, numBatchesXEpoch, "Loss: %.6f | Accuracy: %.4f | Time: %.4f" % (
                    lossAccumValue/10.0, currAccuracy*100.0, (endTrainTime-startTrainTime)/1000.0))

                with open(logFile, "a") as myfile:
                    myfile.write("Step: %6d (%4d) | Loss: %.6f | Accuracy: %.4f\n" % (step, epochStep, lossAccumValue/10.0, currAccuracy*100.0))

                sess.run(resetMetrics)
                lossInfoCounter = 0
                lossAccumValue = 0.0
                startTrainTime = current_milli_time()

            step += 1
            epochStep += 1

        endEpochTime = current_milli_time()   
        print("Epoch %3d  train time: %.4f" %(epoch, (endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch %3d  train time: %.4f \n" %(epoch, (endEpochTime-startEpochTime)/1000.0))

        #Test data
        accumTestLoss = 0.0
        sess.run(resetMetrics)
        IoUxCat = [[] for i in range(len(cat))]
        for i in range(numTestModels):
            currTest = testFiles[i]
            points = []
            batchIds = []
            catLabels = []
            features = []
            labels = []

            catId = 0
            for currCat in range(len(cat)):
                if cat[currCat][1] in currTest:
                    catId = currCat
            
            load_model(currTest, catId, 0, points, batchIds, features, catLabels, labels, 1.0, False, False)
            
            lossRes, predictedLabelsRes, _ = sess.run([loss, predictedLabels, accuracyAccumOp], 
                    {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels, 
                    inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})

            accumTestLoss = accumTestLoss + lossRes
            
            #Compute IoU
            numParts = len(segClasses[cat[catId][0]])
            accumIoU = 0.0
            for j in range(numParts):
                intersection = 0.0
                union = 0.0
                currLabel = segClasses[cat[catId][0]][j]
                for k in range(len(labels)):
                    if labels[k] == predictedLabelsRes[k] and labels[k] == currLabel:
                        intersection = intersection + 1.0
                    if labels[k] == currLabel or predictedLabelsRes[k] == currLabel:
                        union = union + 1.0
                if union > 0.0:
                    accumIoU = accumIoU + intersection/union
                else:
                    accumIoU = accumIoU + 1.0
            accumIoU = accumIoU/float(numParts)
            IoUxCat[catId].append(accumIoU)
            
            if i%100 == 0:
                visualize_progress(i, numTestModels)

        #Compute mean IoU
        meanIoUxCat = 0.0
        for i in range(len(IoUxCat)):
            currMean = 0.0
            for currVal in IoUxCat[i]:
                currMean = currMean + currVal
            currMean = currMean / float(len(IoUxCat[i]))
            print("Mean IoU category "+cat[i][0]+": "+str(currMean))
            meanIoUxCat = meanIoUxCat + currMean*float(len(IoUxCat[i]))
        meanIoUxCat = meanIoUxCat / float(numTestModels)

        totalAccuracy, metricsTestSummRes = sess.run([accuracyVal, metricsTestSummary], {iouVal: meanIoUxCat})
        accumTestLoss = accumTestLoss/float(numTestModels)
        summary_writer.add_summary(metricsTestSummRes, step)

        print("Loss: %.6f | Test accuracy: %.4f | Test IoU %.4f" % (accumTestLoss, totalAccuracy*100.0, meanIoUxCat*100.0))
        with open(logFile, "a") as myfile:
            myfile.write("Loss: %.6f | Test accuracy: %.4f | Test IoU %.4f\n" % (accumTestLoss, totalAccuracy*100.0, meanIoUxCat*100.0))
    
        if epoch%10==0:
            saver.save(sess, args.logFolder+"/model.ckpt")
