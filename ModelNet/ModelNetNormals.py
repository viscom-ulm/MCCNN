'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ModelNet.py

    \brief Code to train a normal estimation network on the ModelNet40 
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

def create_angle(convResult, normals):
    normalized_conv = tf.nn.l2_normalize(convResult, axis=1)
    normalized_normals = tf.nn.l2_normalize(normals, axis=1)
    error = tf.multiply(normalized_conv, normalized_normals)
    error = tf.reduce_sum(error, 1)
    return tf.acos(tf.reduce_mean(error))

def create_loss(convResult, normals): 
    normalized_normals = tf.nn.l2_normalize(normals, axis=1)
    normalized_conv = tf.nn.l2_normalize(convResult, axis=1)
    return tf.losses.cosine_distance(normalized_normals, normalized_conv, axis = 1)

def create_trainning(lossGraph, learningRate, maxLearningRate, learningDecayFactor, learningRateDecay, global_step):
    learningRateExp = tf.train.exponential_decay(learningRate, global_step, learningRateDecay, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph, global_step=global_step)
    return train_op, learningRateExp


def load_model(modelsPath, numPoints, batchId, outPts, outBatchIds, outFeatures, outNormals, augmentData, training):
    with open(modelsPath, 'r') as modelFile:  
        # Matrix random generation to rotate along up direction (Code obtained from PointNet++)
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
        angles = np.clip(0.06*np.random.randn(3), -0.18, 0.18)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        R = np.dot(R, rotation_matrix)

        iter = 0
        for line in modelFile:
            if iter < numPoints:
                line = line.replace("\n", "")
                currPoint = line.split(',')
                auxPoint = np.array([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
                if augmentData and training:
                    auxPoint = np.dot(auxPoint, R)
                    displ = np.clip(0.01 * np.random.randn(3), -0.01, 0.01)
                else:
                    displ = np.array([0.0, 0.0, 0.0])
                outPts.append([auxPoint[0] + displ[0], auxPoint[1] + displ[1], auxPoint[2] + displ[2]])
                outFeatures.append([1])
                outBatchIds.append([batchId])
                outNormals.append([float(currPoint[3]), float(currPoint[4]), float(currPoint[5])])
                iter = iter + 1
            else:
                break

def get_train_and_test_files():
    #Get the category names.
    catNames =[]
    with open("./data/modelnet40_shape_names.txt", 'r') as nameFile:
        for line in nameFile:
            catNames.append(line.replace("\n",""))
    print(catNames)

    #List of files to train
    trainFiles = []
    with open("./data/modelnet40_train.txt", 'r') as nameFile:
        for line in nameFile:
            catId = -1
            for i in range(len(catNames)):
                if catNames[i] in line:
                    catId = i
                    break
            if catId >= 0:
                trainFiles.append("./data/"+catNames[catId]+"/"+line.replace("\n","")+".txt")

    #List of files to test
    testFiles = []
    with open("./data/modelnet40_test.txt", 'r') as nameFile:
        for line in nameFile:
            catId = -1
            for i in range(len(catNames)):
                if catNames[i] in line:
                    catId = i
                    break
            if catId >= 0:
                testFiles.append("./data/"+catNames[catId]+"/"+line.replace("\n","")+".txt")

    return trainFiles, testFiles

def visualize_progress(val, maxVal, description="", barWidth=20):

    progress = int((val*barWidth) / maxVal)
    progressBar = ['='] * (progress) + ['>'] + ['.'] * (barWidth - (progress+1))
    progressBar = ''.join(progressBar)
    initBar = "%4d/%4d" % (val + 1, maxVal)
    print(initBar + ' [' + progressBar + '] ' + description)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train MCCNN for normal estimation of point clouds (ModelNet40)')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--model', default='MCNorm', help='model (default: MCNorm)')
    parser.add_argument('--grow', default=32, type=int, help='Growth rate (default: 32)')
    parser.add_argument('--batchSize', default=16, type=int, help='Batch size  (default: 16)')
    parser.add_argument('--maxEpoch', default=200, type=int, help='Max Epoch  (default: 200)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDeacyFactor', default=0.5, type=float, help='Learning deacy factor (default: 0.5)')
    parser.add_argument('--learningDecayRate', default=20, type=int, help='Learning decay rate  (default: 20 Epochs)')
    parser.add_argument('--maxLearningRate', default=0.00001, type=float, help='Maximum Learning rate (default: 0.00001)')
    parser.add_argument('--nPoints', default=1024, type=int, help='Number of points (default: 1024)')
    parser.add_argument('--augment', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    args = parser.parse_args()

    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp ../models/%s.py %s' % (args.model, args.logFolder))
    os.system('cp ModelNetNormals.py %s' % (args.logFolder))
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
        myFile.write("nPoints: "+str(args.nPoints)+"\n")
        myFile.write("Augment: "+str(args.augment)+"\n")

    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("BatchSize: "+str(args.batchSize))
    print("MaxEpoch: "+str(args.maxEpoch))
    print("InitLearningRate: "+str(args.initLearningRate))
    print("LearningDeacyFactor: "+str(args.learningDeacyFactor))
    print("LearningDecayRate: "+str(args.learningDecayRate))
    print("MaxLearningRate: "+str(args.maxLearningRate))
    print("nPoints: "+str(args.nPoints))
    print("Augment: "+str(args.augment))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    trainFiles, testFiles = get_train_and_test_files()
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
    inNormals = tf.placeholder(tf.float32, [None, 3])
    isTraining = tf.placeholder(tf.bool)
    accuracyAngle = tf.placeholder(tf.float32)
    accuracyTestAngle = tf.placeholder(tf.float32)
    
    #Create the network
    predNormals = model.create_network(inPts, inBatchIds, inFeatures, 1, args.batchSize, args.grow, isTraining)
          
    #Create loss
    loss = create_loss(predNormals, inNormals)

    #Create angle
    angle = create_angle(predNormals, inNormals)

    #Create training
    trainning, learningRateExp = create_trainning(loss, 
        args.initLearningRate, args.maxLearningRate, args.learningDeacyFactor, 
        args.learningDecayRate*numBatchesXEpoch, global_step)
    learningRateSumm = tf.summary.scalar('learninRate', learningRateExp)

    #Create sumaries
    lossSummary = tf.summary.scalar('loss', loss)
    trainingSummary = tf.summary.merge([lossSummary, learningRateSumm])
    metricsSummary = tf.summary.scalar('accuracy', accuracyAngle)
    metricsTestSummary = tf.summary.scalar('Tes_Accuracy', accuracyTestAngle)

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
        angleAccumValue = 0.0

        #Iterate over all the train files
        while len(cpyTrainFiles) > 0:
    
            points = []
            batchIds = []
            features = []
            normals = []
            for i in range(args.batchSize):
                if len(cpyTrainFiles) > 0:
                    currentModelIndex = random.randint(0, len(cpyTrainFiles)-1)

                    currentModel = cpyTrainFiles[currentModelIndex]

                    cpyTrainFiles.pop(currentModelIndex)

                    load_model(currentModel, args.nPoints, i, points, batchIds, 
                        features, normals, args.augment, True)

            _, lossRes, angleRes, trainingSummRes = \
                sess.run([trainning, loss, angle, trainingSummary], 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inNormals: normals, isTraining: True})

            summary_writer.add_summary(trainingSummRes, step)

            angleAccumValue += angleRes
            lossAccumValue += lossRes
            lossInfoCounter += 1

            if lossInfoCounter == 10:
                currAngle = math.degrees(angleAccumValue/10.0)
                endTrainTime = current_milli_time()                   
                metricsSummRes = sess.run(metricsSummary, {accuracyAngle: currAngle})
                summary_writer.add_summary(metricsSummRes, step)

                visualize_progress(epochStep, numBatchesXEpoch, "Loss: %.6f | Angle: %.4f | Time: %.4f" % (
                    lossAccumValue/10.0, currAngle, (endTrainTime-startTrainTime)/1000.0))

                with open(logFile, "a") as myfile:
                    myfile.write("Step: %6d (%4d) | Loss: %.6f | Angle: %.4f\n" % (step, epochStep, lossAccumValue/10.0, currAngle))

                lossInfoCounter = 0
                lossAccumValue = 0.0
                angleAccumValue = 0.0
                startTrainTime = current_milli_time()

            step += 1
            epochStep += 1

        endEpochTime = current_milli_time()   
        print("Epoch %3d  train time: %.4f" %(epoch, (endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch %3d  train time: %.4f \n" %(epoch, (endEpochTime-startEpochTime)/1000.0))

        #Test data
        accumTestLoss = 0.0
        accumAngleTest = 0.0
        for i in range(numTestModels):
            currTest = testFiles[i]
            points = []
            batchIds = []
            features = []
            normals = []

            load_model(currTest, args.nPoints, 0, points, batchIds, features, normals, False, False)
            lossRes, angleRes = sess.run([loss, angle], 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inNormals: normals, isTraining: False})

            accumTestLoss +=lossRes
            accumAngleTest += angleRes
            
            if i%100 == 0:
                visualize_progress(i, numTestModels)

        accumTestLoss = accumTestLoss/float(numTestModels)
        currTestAngle = math.degrees(accumAngleTest/float(numTestModels))
        metricsTestSummRes = sess.run(metricsTestSummary, {accuracyTestAngle: currTestAngle})
        summary_writer.add_summary(metricsTestSummRes, step)

        print("Loss: %.6f | Test accuracy: %.4f" % (accumTestLoss, currTestAngle))
        with open(logFile, "a") as myfile:
            myfile.write("Loss: %.6f | Test accuracy: %.4f\n" % (accumTestLoss, currTestAngle))
    
        if epoch%10==0:
            saver.save(sess, args.logFolder+"/model.ckpt")
