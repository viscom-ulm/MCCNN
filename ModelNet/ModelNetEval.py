'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ModelNet.py

    \brief Code to evaluate a classification network on the ModelNet40 
           dataset.

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
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))

current_milli_time = lambda: time.time() * 1000.0


def create_accuracy(logits, labels, scope):
    _, logitsIndexs = tf.nn.top_k(logits)
    with tf.variable_scope(scope):
        return tf.metrics.accuracy(tf.reshape(labels, [-1, 1]), logitsIndexs)


def load_model(modelsPath, numPoints, catId, batchId, outPts, outBatchIds, outFeatures, outLabels):
    with open(modelsPath, 'r') as modelFile:        
        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                outFeatures.append([1])
                outBatchIds.append([batchId])
                iter = iter + 1
            else:
                break

    outLabels.append(catId)
    
def load_model_non_uniform_gradient(modelsPath, numPoints, catId, batchId, outPts, outBatchIds, outFeatures, outLabels):
    with open(modelsPath, 'r') as modelFile:
        maxPt = np.array([-10000.0, -10000.0, -10000.0])
        minPt = np.array([10000.0, 10000.0, 10000.0])
        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                maxPt = np.maximum(auxPoint, maxPt)
                minPt = np.minimum(auxPoint, minPt)
                iter = iter + 1
            else:
                break
        modelSize = maxPt - minPt
        largAxis = 0
        if modelSize[1] > modelSize[0]:
            largAxis = 1
        if modelSize[2] > modelSize[largAxis]:
            largAxis = 2

    with open(modelsPath, 'r') as modelFile:
        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                rndNum = np.random.random()
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                probVal = (auxPoint[largAxis]-minPt[largAxis]-modelSize[largAxis]*0.2)/(modelSize[largAxis]*0.6)
                keepProbModif = pow(np.clip(probVal, 0.01, 1.0), 1.0/2.0)
                if rndNum < keepProbModif:
                    outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                    outFeatures.append([1])
                    outBatchIds.append([batchId])
                    iter = iter +1
            else:
                break
    outLabels.append(catId)
    

def load_model_non_uniform_lambert(modelsPath, numPoints, catId, batchId, outPts, outBatchIds, outFeatures, outLabels, viewDir):
    with open(modelsPath, 'r') as modelFile:
        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                auxNormal = np.array([float(currPoint[3]), float(currPoint[4]), float(currPoint[5])])
                dotVal = np.dot(viewDir, auxNormal)
                dotVal = pow(np.clip(dotVal, 0.0, 1.0), 0.5)
                rndNum = np.random.random()
                if rndNum < dotVal:
                    outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                    outFeatures.append([1])
                    outBatchIds.append([batchId])     
                    iter = iter +1 

    outLabels.append(catId)

def load_model_non_uniform_split(modelsPath, numPoints, catId, batchId, outPts, outBatchIds, outFeatures, outLabels, probPointVal):
    with open(modelsPath, 'r') as modelFile:
        maxPt = np.array([-10000.0, -10000.0, -10000.0])
        minPt = np.array([10000.0, 10000.0, 10000.0])
        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                maxPt = np.maximum(auxPoint, maxPt)
                minPt = np.minimum(auxPoint, minPt)
                iter = iter + 1
            else:
                break
        modelSize = maxPt - minPt
        largAxis = 0
        if modelSize[1] > modelSize[0]:
            largAxis = 1
        if modelSize[2] > modelSize[largAxis]:
            largAxis = 2

    with open(modelsPath, 'r') as modelFile:
        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                rndNum = np.random.random()
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                probVal = (auxPoint[largAxis]-minPt[largAxis])/modelSize[largAxis]
                if probVal > 0.5:
                    probVal = 1.0
                else:
                    probVal = probPointVal
                if rndNum < probVal:
                    outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                    outFeatures.append([1])
                    outBatchIds.append([batchId])
                    iter = iter +1

    outLabels.append(catId)


def get_train_and_test_files():
    #Get the category names.
    catNames =[]
    with open("./data/modelnet40_shape_names.txt", 'r') as nameFile:
        for line in nameFile:
            catNames.append(line.replace("\n",""))
    print(catNames)

    #List of files to train
    trainFiles = []
    trainCatFileId = []
    with open("./data/modelnet40_train.txt", 'r') as nameFile:
        for line in nameFile:
            catId = -1
            for i in range(len(catNames)):
                if catNames[i] in line:
                    catId = i
                    break
            if catId >= 0:
                trainFiles.append("./data/"+catNames[catId]+"/"+line.replace("\n","")+".txt")
                trainCatFileId.append(catId)

    #List of files to test
    testFiles = []
    testCatFileId = []
    with open("./data/modelnet40_test.txt", 'r') as nameFile:
        for line in nameFile:
            catId = -1
            for i in range(len(catNames)):
                if catNames[i] in line:
                    catId = i
                    break
            if catId >= 0:
                testFiles.append("./data/"+catNames[catId]+"/"+line.replace("\n","")+".txt")
                testCatFileId.append(catId)

    return catNames, trainFiles, trainCatFileId, testFiles, testCatFileId

def visualize_progress(val, maxVal, description="", barWidth=20):

    progress = int((val*barWidth) / maxVal)
    progressBar = ['='] * (progress) + ['>'] + ['.'] * (barWidth - (progress+1))
    progressBar = ''.join(progressBar)
    initBar = "%4d/%4d" % (val + 1, maxVal)
    print(initBar + ' [' + progressBar + '] ' + description)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation of classification networks (ModelNet40)')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--model', default='MCClassSmall', help='model (default: MCClassSmall)')
    parser.add_argument('--grow', default=64, type=int, help='Growth rate (default: 64)')
    parser.add_argument('--nPoints', default=1024, type=int, help='Number of points (default: 1024)')
    parser.add_argument('--nExec', default=5, type=int, help='Number of executions per model (default: 5)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    args = parser.parse_args()

    print("Trained model: "+args.inTrainedModel)
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("nPoints: "+str(args.nPoints))
    
    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    catNames, trainFiles, trainCatFileId, testFiles, testCatFileId = get_train_and_test_files()
    numTestModels = len(testFiles)
    print("Test models: " + str(numTestModels))

    #Create variable and place holders
    inPts = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None, 1])
    inFeatures = tf.placeholder(tf.float32, [None, 1])
    inLabels = tf.placeholder(tf.int32, [None])
    isTraining = tf.placeholder(tf.bool, shape=())
    keepProbConv = tf.placeholder(tf.float32)
    keepProbFull = tf.placeholder(tf.float32)

    #Create the network
    logits = model.create_network(inPts, inBatchIds, inFeatures, 1, 1, args.grow, len(catNames), isTraining, keepProbFull, keepProbConv, False, False)
          
    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver()
    
    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #Init variables
    sess.run(init)
    sess.run(initLocal)

    #Restore the model
    saver.restore(sess, args.inTrainedModel)
    
    #Test data uniform sampling
    print("Uniform sampling")
 
    accumTime = 0.0
    totalAccuracy = 0.0
    for i in range(numTestModels):
        currTest = testFiles[i]
        currCatId = testCatFileId[i]
        points = []
        batchIds = []
        features = []
        labels = []

        load_model(currTest, args.nPoints, currCatId, 0, points, batchIds, features, labels)
        logitsSum = np.zeros((1, len(catNames)))
        for j in range(args.nExec):
            startTimeMeasure = current_milli_time() 
            logitsRes = sess.run(logits, 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            endTimeMeasure = current_milli_time() 
            accumTime = accumTime + (endTimeMeasure - startTimeMeasure)
            logitsSum += logitsRes
            if (i*args.nExec + j)%100 == 0:
                visualize_progress(i*args.nExec+j, numTestModels*args.nExec)
        
        predCat = np.argmax(logitsSum, 1)
        if predCat == currCatId:
            totalAccuracy = totalAccuracy + 1.0
            
    print("Time: %.8f" % (accumTime/(float(numTestModels)*float(args.nExec))))

    totalAccuracy = totalAccuracy/float(numTestModels)
    print("Test accuracy: %.4f" % (totalAccuracy*100.0))
    
    #Test data non-uniform gradient sampling
    print("Non-Uniform sampling gradient")
    totalAccuracy = 0.0
    for i in range(numTestModels):
        currTest = testFiles[i]
        currCatId = testCatFileId[i]
       
        logitsSum = np.zeros((1, len(catNames)))
        for j in range(args.nExec):
            points = []
            batchIds = []
            features = []
            labels = []

            load_model_non_uniform_gradient(currTest, args.nPoints, currCatId, 0, points, batchIds, features, labels)
            logitsRes = sess.run(logits, 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            logitsSum += logitsRes
            if (i*args.nExec + j)%100 == 0:
                visualize_progress(i*args.nExec+j, numTestModels*args.nExec)
        
        predCat = np.argmax(logitsSum, 1)
        if predCat == currCatId:
            totalAccuracy = totalAccuracy + 1.0

    totalAccuracy = totalAccuracy/float(numTestModels)
    print("Test accuracy: %.4f" % (totalAccuracy*100.0))
    
    #Test data non-uniform gradient lambert
    print("Non-Uniform sampling lambert")
    totalAccuracy = 0.0
    for i in range(numTestModels):
        currTest = testFiles[i]
        currCatId = testCatFileId[i]

        logitsSum = np.zeros((1, len(catNames)))
        for j in range(args.nExec):
            points = []
            batchIds = []
            features = []
            labels = []

            auxView = (np.random.rand(3)*2.0)-1.0
            auxView = auxView / np.linalg.norm(auxView)
            load_model_non_uniform_lambert(currTest, args.nPoints, currCatId, 0, points, batchIds, features, labels, auxView)
            logitsRes = sess.run(logits,
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            logitsSum += logitsRes
            if (i*args.nExec + j)%100 == 0:
                visualize_progress(i*args.nExec+j, numTestModels*args.nExec)
            
        predCat = np.argmax(logitsSum, 1)
        if predCat == currCatId:
            totalAccuracy = totalAccuracy + 1.0

    totalAccuracy = totalAccuracy/float(numTestModels)
    print("Test accuracy: %.4f" % (totalAccuracy*100.0))
    
    #Test data non-uniform gradient split
    print("Non-Uniform sampling split")
    totalAccuracy = 0.0
    for i in range(numTestModels):
        currTest = testFiles[i]
        currCatId = testCatFileId[i]

        logitsSum = np.zeros((1, len(catNames)))
        for j in range(args.nExec):
            points = []
            batchIds = []
            features = []
            labels = []

            load_model_non_uniform_split(currTest, args.nPoints, currCatId, 0, points, batchIds, features, labels, 0.25)
            logitsRes = sess.run(logits,
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            logitsSum += logitsRes
            if (i*args.nExec + j)%100 == 0:
                visualize_progress(i*args.nExec+j, numTestModels*args.nExec)
            
        predCat = np.argmax(logitsSum, 1)
        if predCat == currCatId:
            totalAccuracy = totalAccuracy + 1.0

    totalAccuracy = totalAccuracy/float(numTestModels)
    print("Test accuracy: %.4f" % (totalAccuracy*100.0))
