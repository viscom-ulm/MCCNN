'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ShapeNet.py

    \brief Code to evaluate a segmentation network on the ShapeNet dataset.

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


def create_accuracy(logits, labels, scope):
    _, logitsIndexs = tf.nn.top_k(logits)
    with tf.variable_scope(scope):
        return tf.metrics.accuracy(tf.reshape(labels, [-1, 1]), logitsIndexs)


def load_model(modelsPath, catId, batchId, outPts, outBatchIds, outFeatures, outCatLabels, outLabels):
    with open(modelsPath+".txt", 'r') as modelFile:
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split()
            auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
            outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
            outFeatures.append([1])
            outLabels.append(int(float(currPoint[6])))
            outCatLabels.append([catId])
            outBatchIds.append([batchId])

def load_model_non_uniform_gradient(modelsPath, catId, batchId, outPts, outBatchIds, outFeatures, outCatLabels, outLabels):
    with open(modelsPath+".txt", 'r') as modelFile:
        maxPt = np.array([-10000.0, -10000.0, -10000.0])
        minPt = np.array([10000.0, 10000.0, 10000.0])
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split()
            auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
            maxPt = np.maximum(auxPoint, maxPt)
            minPt = np.minimum(auxPoint, minPt)
        modelSize = maxPt - minPt
        largAxis = 0
        if modelSize[1] > modelSize[0]:
            largAxis = 1
        if modelSize[2] > modelSize[largAxis]:
            largAxis = 2

        with open(modelsPath+".txt", 'r') as modelFile:
            probs = []
            for line in modelFile:
                line = line.replace("\n", "")
                currPoint = line.split()
                rndNum = np.random.random()
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                probVal = (auxPoint[largAxis]-minPt[largAxis]-modelSize[largAxis]*0.2)/(modelSize[largAxis]*0.6)
                keepProbModif = pow(np.clip(probVal, 0.01, 1.0), 1.0/2.0)
                if rndNum < keepProbModif:
                    probs.append(keepProbModif)
                    outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                    outFeatures.append([1])
                    outLabels.append(int(float(currPoint[6])))
                    outCatLabels.append([catId])
                    outBatchIds.append([batchId])
                        
def load_model_non_uniform_lambert(modelsPath, catId, batchId, outPts, outBatchIds, outFeatures, outCatLabels, outLabels, viewDir):
    with open(modelsPath+".txt", 'r') as modelFile:
        maxPt = np.array([-10000.0, -10000.0, -10000.0])
        minPt = np.array([10000.0, 10000.0, 10000.0])
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split()
            auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
            auxNormal = np.array([float(currPoint[3]), float(currPoint[4]), float(currPoint[5])])
            dotVal = np.dot(viewDir, auxNormal)
            dotVal = pow(np.clip(dotVal, 0.0, 1.0), 0.5)
            rndNum = np.random.random()
            if rndNum < dotVal:
                maxPt = np.maximum(auxPoint, maxPt)
                minPt = np.minimum(auxPoint, minPt)
                outPts.append([auxPoint[0], auxPoint[1], auxPoint[2]])
                outFeatures.append([1])
                outLabels.append(int(float(currPoint[6])))
                outCatLabels.append([catId])
                outBatchIds.append([batchId])
                    

def load_model_non_uniform_split(modelsPath, catId, batchId, outPts, outBatchIds, outFeatures, outCatLabels, outLabels, probPointVal):
    with open(modelsPath+".txt", 'r') as modelFile:
        maxPt = np.array([-10000.0, -10000.0, -10000.0])
        minPt = np.array([10000.0, 10000.0, 10000.0])
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split()
            auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
            maxPt = np.maximum(auxPoint, maxPt)
            minPt = np.minimum(auxPoint, minPt)
        modelSize = maxPt - minPt
        largAxis = 0
        if modelSize[1] > modelSize[0]:
            largAxis = 1
        if modelSize[2] > modelSize[largAxis]:
            largAxis = 2


    with open(modelsPath+".txt", 'r') as modelFile:
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split()
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
                outLabels.append(int(float(currPoint[6])))
                outCatLabels.append([catId])
                outBatchIds.append([batchId])
                    
def saveModelWithLabels(modelName, points, labels): 
    with open(modelName+".ply", 'w') as myFile:
        myFile.write("ply\n")
        myFile.write("format ascii 1.0\n")
        myFile.write("element vertex "+ str(len(points))+"\n")
        myFile.write("property float x\n")
        myFile.write("property float y\n")
        myFile.write("property float z\n")
        myFile.write("property uchar red\n")
        myFile.write("property uchar green\n")
        myFile.write("property uchar blue\n")
        myFile.write("end_header\n")

        for point, label in zip(points, labels):
            pointFlt = [float(point[0]), float(point[1]), float(point[2])]
            currLabel = label%6
            color = [0, 0, 0]
            if currLabel == 0:
                color = [228,26,28]
            elif currLabel == 1:
                color = [55,126,184]
            elif currLabel == 2:
                color = [77,175,74]
            elif currLabel == 3:
                color = [152,78,163]
            elif currLabel == 4:
                color = [255,127,0]
            else:
                color = [255,255,51]
            myFile.write(str(pointFlt[0])+" "+str(pointFlt[1])+" "+str(pointFlt[2])+" "+str(color[0])+" "+ str(color[1])+ " "+str(color[2])+"\n")
        
    myFile.close()

    
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

    parser = argparse.ArgumentParser(description='Evaluation of segmentation networks (ShapeNet)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--model', default='MCSeg', help='model (default: MCSeg)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    parser.add_argument('--nExec', default=5, type=int, help='Number of executions per model (default: 5)')
    parser.add_argument('--saveModels', action='store_true', help='Save models (default: False)')
    args = parser.parse_args()


    print("Trained model: "+args.inTrainedModel)
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    cat, segClasses, trainFiles, testFiles = get_train_and_test_files()
    numTestModels = len(testFiles)
    print("Test models: " + str(numTestModels))
    
    #Save models, create folder
    if args.saveModels:
        if not os.path.exists("savedModels"): os.mkdir("savedModels")

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

    #Create the network
    logits = model.create_network(inPts, inBatchIds, inFeatures,inCatLabels, 1, len(cat), 50, 1, 
        args.grow, isTraining, keepProbConv, keepProbFull, False, False)
          
    #Create predict labels
    predictedLabels = tf.argmax(logits, 1)
    
    #Create accuracy metric
    accuracyVal, accuracyAccumOp = create_accuracy(logits, inLabels, 'metrics')
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

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

    step = 0
    epochStep = 0
    np.random.seed(int(time.time()))
    

    #Test data
    print("Uniform sampling")
    accumTime = 0.0
    sess.run(resetMetrics)
    IoUxCat = [[] for i in range(len(cat))]
    for i in range(numTestModels):
        currTest = testFiles[i]
        for ex in range(args.nExec):
            points = []
            batchIds = []
            catLabels = []
            features = []
            labels = []
            catId = 0
            for currCat in range(len(cat)):
                if cat[currCat][1] in currTest:
                    catId = currCat
           
            load_model(currTest, catId, 0, points, batchIds, features, catLabels, labels)
           
            startTimeMeasure = current_milli_time()
            predictedLabelsRes, _ = sess.run([predictedLabels, accuracyAccumOp], 
                    {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels, 
                    inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            endTimeMeasure = current_milli_time() 
            accumTime = accumTime + (endTimeMeasure - startTimeMeasure)
            
            #Save models
            if args.saveModels:
                rndNum = np.random.random()
                if rndNum < 0.1:
                    saveModelWithLabels("savedModels/"+currTest.replace("/", "-")+"_gt", points, labels)
                    saveModelWithLabels("savedModels/"+currTest.replace("/", "-")+"_pred", points, predictedLabelsRes)
            
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
            
            if ((i*args.nExec)+ex) % 100 == 0:
                visualize_progress((i*args.nExec)+ex, numTestModels*args.nExec)

    #Compute mean IoU
    meanIoUxCat = 0.0
    for i in range(len(IoUxCat)):
        currMean = 0.0
        for currVal in IoUxCat[i]:
            currMean = currMean + currVal
        currMean = currMean / float(len(IoUxCat[i]))
        print("Mean IoU category "+cat[i][0]+": "+str(currMean))
        meanIoUxCat = meanIoUxCat + currMean*float(len(IoUxCat[i]))
    meanIoUxCat = meanIoUxCat / float(numTestModels*args.nExec)

    totalAccuracy = sess.run(accuracyVal)

    print("Time: %.8f" % (accumTime/(float(numTestModels)*float(args.nExec))))
    print("Test accuracy: %.4f | Test IoU %.4f" % (totalAccuracy*100.0, meanIoUxCat*100.0))

    #Test data
    print("Gradient sampling")
    sess.run(resetMetrics)
    IoUxCat = [[] for i in range(len(cat))]
    for i in range(numTestModels):
        currTest = testFiles[i]
        for ex in range(args.nExec):
            points = []
            batchIds = []
            catLabels = []
            features = []
            labels = []
            catId = 0
            for currCat in range(len(cat)):
                if cat[currCat][1] in currTest:
                    catId = currCat
           
            load_model_non_uniform_gradient(currTest, catId, 0, points, batchIds, features, catLabels, labels)
           
            predictedLabelsRes, _ = sess.run([predictedLabels, accuracyAccumOp], 
                    {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels, 
                    inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})

            
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
            
            if ((i*args.nExec)+ex) % 100 == 0:
                visualize_progress((i*args.nExec)+ex, numTestModels*args.nExec)

    #Compute mean IoU
    meanIoUxCat = 0.0
    for i in range(len(IoUxCat)):
        currMean = 0.0
        for currVal in IoUxCat[i]:
            currMean = currMean + currVal
        currMean = currMean / float(len(IoUxCat[i]))
        print("Mean IoU category "+cat[i][0]+": "+str(currMean))
        meanIoUxCat = meanIoUxCat + currMean*float(len(IoUxCat[i]))
    meanIoUxCat = meanIoUxCat / float(numTestModels*args.nExec)

    totalAccuracy = sess.run(accuracyVal)

    print("Test accuracy: %.4f | Test IoU %.4f" % (totalAccuracy*100.0, meanIoUxCat*100.0))

    
    #Test data
    print("Lambert sampling")
    sess.run(resetMetrics)
    IoUxCat = [[] for i in range(len(cat))]
    for i in range(numTestModels):
        currTest = testFiles[i]
        for ex in range(args.nExec):
            points = []
            batchIds = []
            catLabels = []
            features = []
            labels = []
            catId = 0
            for currCat in range(len(cat)):
                if cat[currCat][1] in currTest:
                    catId = currCat

            auxView = (np.random.rand(3)*2.0)-1.0
            auxView = auxView / np.linalg.norm(auxView)
            load_model_non_uniform_lambert(currTest, catId, 0, points, batchIds, features, catLabels, labels, auxView)

            predictedLabelsRes, _ = sess.run([predictedLabels, accuracyAccumOp],
                    {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels,
                    inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})


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

            if ((i*args.nExec)+ex) % 100 == 0:
                visualize_progress((i*args.nExec)+ex, numTestModels*args.nExec)

    #Compute mean IoU
    meanIoUxCat = 0.0
    for i in range(len(IoUxCat)):
        currMean = 0.0
        for currVal in IoUxCat[i]:
            currMean = currMean + currVal
        currMean = currMean / float(len(IoUxCat[i]))
        print("Mean IoU category "+cat[i][0]+": "+str(currMean))
        meanIoUxCat = meanIoUxCat + currMean*float(len(IoUxCat[i]))
    meanIoUxCat = meanIoUxCat / float(numTestModels*args.nExec)

    totalAccuracy = sess.run(accuracyVal)

    print("Test accuracy: %.4f | Test IoU %.4f" % (totalAccuracy*100.0, meanIoUxCat*100.0))


    #Test data
    print("Split sampling")
    sess.run(resetMetrics)
    IoUxCat = [[] for i in range(len(cat))]
    for i in range(numTestModels):
        currTest = testFiles[i]
        for ex in range(args.nExec):
            points = []
            batchIds = []
            catLabels = []
            features = []
            labels = []
            catId = 0
            for currCat in range(len(cat)):
                if cat[currCat][1] in currTest:
                    catId = currCat

            load_model_non_uniform_split(currTest, catId, 0, points, batchIds, features, catLabels, labels, 0.25)

            predictedLabelsRes, _ = sess.run([predictedLabels, accuracyAccumOp],
                    {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels,
                    inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})


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

            if ((i*args.nExec)+ex) % 100 == 0:
                visualize_progress((i*args.nExec)+ex, numTestModels*args.nExec)

    #Compute mean IoU
    meanIoUxCat = 0.0
    for i in range(len(IoUxCat)):
        currMean = 0.0
        for currVal in IoUxCat[i]:
            currMean = currMean + currVal
        currMean = currMean / float(len(IoUxCat[i]))
        print("Mean IoU category "+cat[i][0]+": "+str(currMean))
        meanIoUxCat = meanIoUxCat + currMean*float(len(IoUxCat[i]))
    meanIoUxCat = meanIoUxCat / float(numTestModels*args.nExec)

    totalAccuracy = sess.run(accuracyVal)

    print("Test accuracy: %.4f | Test IoU %.4f" % (totalAccuracy*100.0, meanIoUxCat*100.0))

