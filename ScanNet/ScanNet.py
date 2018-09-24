'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file genScanNetData.py

    \brief Code to train a segmentation network on the ScanNet dataset.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import math
import time
import argparse
import importlib
import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from PyUtils import visualize_progress
from ScanNetDataSet import ScanNetDataSet

current_milli_time = lambda: time.time() * 1000.0


def create_loss(logits, labels, labelWeights, weigthDecay):
    labels = tf.to_int64(labels)
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        labels=tf.reshape(labels, [-1]), logits=logits, 
        weights=tf.reshape(labelWeights, [-1]), scope='xentropy')
    xentropyloss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    regularizer = tf.contrib.layers.l2_regularizer(scale=weigthDecay)
    regVariables = tf.get_collection('weight_decay_loss')
    regTerm = tf.contrib.layers.apply_regularization(regularizer, regVariables)
    return xentropyloss, regTerm


def create_accuracy(logits, labels, inAccWeights, scope):
    _, logitsIndexs = tf.nn.top_k(logits)
    with tf.variable_scope(scope):
        return tf.metrics.accuracy(labels, logitsIndexs, weights=inAccWeights)


def create_trainning(lossGraph, learningRate, maxLearningRate, learningDecayFactor, learningRateDecay, epoch_step):
    learningRateExp = tf.train.exponential_decay(learningRate, epoch_step, learningRateDecay, learningDecayFactor, staircase=True)
    learningRateExp = tf.maximum(learningRateExp, maxLearningRate)
    optimizer = tf.train.AdamOptimizer(learning_rate =learningRateExp)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(lossGraph)
    return train_op, learningRateExp

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train MCCNN for segmentation tasks (S3DIS)')
    parser.add_argument('--logFolder', default='log', help='Folder of the output models (default: log)')
    parser.add_argument('--model', default='MCSegScanNet', help='model (default: MCSegScanNet)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--batchSize', default=4, type=int, help='Batch size  (default: 4)')
    parser.add_argument('--maxEpoch', default=201, type=int, help='Max Epoch  (default: 201)')
    parser.add_argument('--initLearningRate', default=0.005, type=float, help='Init learning rate  (default: 0.005)')
    parser.add_argument('--learningDeacyFactor', default=0.5, type=float, help='Learning deacy factor (default: 0.5)')
    parser.add_argument('--learningDecayRate', default=20, type=int, help='Learning decay rate  (default: 20 Epochs)')
    parser.add_argument('--maxLearningRate', default=0.00001, type=float, help='Maximum Learning rate (default: 0.00001)')
    parser.add_argument('--useDropOut', action='store_true', help='Use drop out  (default: True)')
    parser.add_argument('--dropOutKeepProb', default=0.5, type=float, help='Keep neuron probabillity drop out  (default: 0.5)')
    parser.add_argument('--useDropOutConv', action='store_true', help='Use drop out in convolution layers (default: False)')
    parser.add_argument('--dropOutKeepProbConv', default=0.8, type=float, help='Keep neuron probabillity drop out in convolution layers (default: 0.8)')
    parser.add_argument('--weightDecay', default=0.0, type=float, help='Weight decay (default: 0.0)')
    parser.add_argument('--maxNumPts', default=600000, type=int, help='Maximum number of points (default: 600000)')
    parser.add_argument('--ptDropOut', default=0.85, type=float, help='Point drop out (default: 0.85)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=1.0, type=float, help='GPU memory used (default: 1.0)')
    parser.add_argument('--augment', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--useColor', action='store_true', help='Augment data (default: False)')
    args = parser.parse_args()

    #Create log folder.
    if not os.path.exists(args.logFolder): os.mkdir(args.logFolder)
    os.system('cp ../models/%s.py %s' % (args.model, args.logFolder))
    os.system('cp ScanNet.py %s' % (args.logFolder))
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
        myFile.write("MaxNumPts: "+str(args.maxNumPts)+"\n")
        myFile.write("ptDropOut: "+str(args.ptDropOut)+"\n")
        myFile.write("Augment: "+str(args.augment)+"\n")
        myFile.write("Use color: "+str(args.useColor)+"\n")

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
    print("MaxNumPts: "+str(args.maxNumPts))
    print("ptDropOut: "+str(args.ptDropOut))
    print("Augment: "+str(args.augment))
    print("Use color: "+str(args.useColor))

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    mTrainDataSet = ScanNetDataSet(0, args.batchSize, args.ptDropOut,
        args.maxNumPts, args.augment, args.useColor)
    mTestDataSet = ScanNetDataSet(1, 1, 1.0, 0, False, args.useColor)
    semLabels = mTrainDataSet.get_labels()
    print(semLabels)
    numTrainRooms = mTrainDataSet.get_num_models()
    numTestRooms = mTestDataSet.get_num_models()
    print("Num train rooms: " + str(numTrainRooms))
    print("Num test rooms: " + str(numTestRooms))
    
    #Create variable and place holders
    epoch_step = tf.Variable(0, name='epoch_step', trainable=False)
    inPts = tf.placeholder(tf.float32, [None, 3])
    inBatchIds = tf.placeholder(tf.int32, [None, 1])
    if args.useColor:
        inFeatures = tf.placeholder(tf.float32, [None, 4])
    else:
        inFeatures = tf.placeholder(tf.float32, [None, 1])
    inLabels = tf.placeholder(tf.int32, [None, 1])
    inWeights = tf.placeholder(tf.float32, [None, 1])
    inAccWeights = tf.placeholder(tf.float32, [None, 1])
    isTraining = tf.placeholder(tf.bool)
    keepProbConv = tf.placeholder(tf.float32)
    keepProbFull = tf.placeholder(tf.float32)
    #Accuracy placeholders
    iouVal = tf.placeholder(tf.float32)
    accuracyVal = tf.placeholder(tf.float32)
    voxelAccuracyVal = tf.placeholder(tf.float32)

    #Increment epoch step
    increment_epoch_step_op = tf.assign(epoch_step, epoch_step+1)

    #Create the network
    numInputs = 1
    if args.useColor:
        numInputs =  4
    logits = model.create_network(inPts, inBatchIds, inFeatures, numInputs, len(semLabels), args.batchSize, 
        args.grow, isTraining, keepProbConv, keepProbFull, args.useDropOutConv, args.useDropOut)
          
    #Create predict labels
    predictedLabels = tf.argmax(logits, 1)
    
    #Create loss
    xentropyLoss, regularizationLoss = create_loss(logits, inLabels, inWeights, args.weightDecay)
    loss = xentropyLoss + regularizationLoss

    #Create training
    trainning, learningRateExp = create_trainning(loss, 
        args.initLearningRate, args.maxLearningRate, args.learningDeacyFactor, 
        args.learningDecayRate, epoch_step)
    learningRateSumm = tf.summary.scalar('learninRate', learningRateExp)

    #Create accuracy metric
    accuracyVal, accuracyAccumOp = create_accuracy(logits, inLabels, inAccWeights, 'metrics')
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

    #Create sumaries
    lossSummary = tf.summary.scalar('loss', loss)
    xEntropyLossSummary = tf.summary.scalar('loss_XEntropy', xentropyLoss)
    regularizationLossSummary = tf.summary.scalar('loss_Regularization', regularizationLoss)
    trainingSummary = tf.summary.merge([lossSummary, xEntropyLossSummary, regularizationLossSummary, learningRateSumm])
    metricsSummary = tf.summary.scalar('accuracy', accuracyVal)
    metricsTestSummary = tf.summary.merge([tf.summary.scalar('Test_Accuracy', accuracyVal), tf.summary.scalar('Test_Voxel_Accuracy', voxelAccuracyVal),
        tf.summary.scalar('Test_IoU', iouVal)], name='TestMetrics')

    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #create the saver
    saver = tf.train.Saver(max_to_keep=100)
    
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
    maxIoU = 0.0
    maxAccuracy = 0.0
    maxVoxAccuracy = 0.0
    maxNoMeantotalAccuracy = 0.0
    np.random.seed(int(time.time()))
    
    #Train
    for epoch in range(args.maxEpoch):
        print("############################## Epoch %3d training" %(epoch))

        startEpochTime = current_milli_time()
        startTrainTime = current_milli_time()

        epochStep = 0

        sess.run(resetMetrics)

        #Iterate over all the train files
        currRoomITer = 0
        mTrainDataSet.start_iteration()
        while mTrainDataSet.has_more_batches():

            modelsxBatch, trainPoints, trainBatchIds, trainFeatures, trainLabels, _, paths = mTrainDataSet.get_next_batch()
            trainWeights = mTrainDataSet.get_weights(trainLabels)
            trainAccWeights = mTrainDataSet.get_accuracy_masks(trainLabels)
        
            currRoomITer += modelsxBatch

            startProcessTime = current_milli_time()
            _, lossRes, xentropyLossRes, regularizationLossRes, trainingSummRes, _ = \
                sess.run([trainning, loss, xentropyLoss, regularizationLoss, trainingSummary, accuracyAccumOp], 
                {inPts: trainPoints, inBatchIds: trainBatchIds, inFeatures: trainFeatures, inWeights: trainWeights, 
                inAccWeights: trainAccWeights, inLabels: trainLabels, isTraining: True, keepProbConv: args.dropOutKeepProbConv, 
                keepProbFull: args.dropOutKeepProb})
            endProcessTime = current_milli_time()   

            summary_writer.add_summary(trainingSummRes, step)

            endTrainTime = current_milli_time()                   
            currAccuracy, metricsSummRes = sess.run([accuracyVal, metricsSummary])
            summary_writer.add_summary(metricsSummRes, step)

            visualize_progress(currRoomITer-1, numTrainRooms, "Loss: %.6f | Accuracy: %.4f | Time: %.4f [%.4f] | Num points: %d" % (
                lossRes, currAccuracy*100.0, (endTrainTime-startTrainTime)/1000.0, (endProcessTime-startProcessTime), len(trainPoints)))

            with open(logFile, "a") as myfile:
                myfile.write("Step: %6d (%4d) | Loss: %.6f | Accuracy: %.4f | Num points: %d\n" % (
                    step, epochStep, lossRes, currAccuracy*100.0, len(trainPoints)))

            sess.run(resetMetrics)
            startTrainTime = current_milli_time()

            step += 1
            epochStep += 1

        endEpochTime = current_milli_time()   
        print("Epoch %3d  train time: %.4f" %(epoch, (endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch %3d  train time: %.4f \n" %(epoch, (endEpochTime-startEpochTime)/1000.0))

        if epoch%10==0:
            saver.save(sess, args.logFolder+"/check_model.ckpt", global_step=epoch)

        #Test data
        print("############################## Epoch %3d evaluation" %(epoch))
        it = 0
        accumLoss = 0.0
        accumTestLoss = 0.0
        sess.run(resetMetrics)
        accumIntersection = [0.0 for i in range(len(semLabels))]
        accumUnion = [0.0 for i in range(len(semLabels))]
        accumGt = [0.0 for i in range(len(semLabels))]
        accumVox = [0.0 for i in range(len(semLabels))]
        accumVoxGt = [0.0 for i in range(len(semLabels))]
        mTestDataSet.start_iteration()
        while mTestDataSet.has_more_batches():

            _, points, batchIds, features, labels, _, _ = mTestDataSet.get_next_batch()
            currAccWeights = mTestDataSet.get_accuracy_masks(labels)
            
            lossRes, predictedLabelsRes = sess.run([loss, predictedLabels], 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inWeights: currAccWeights, 
                inAccWeights: currAccWeights, inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})

            labels = labels.reshape((-1))
            
            #Compute IoU
            for k in range(len(predictedLabelsRes)):
                if labels[k] != 0:
                    if labels[k] == predictedLabelsRes[k]:
                        accumIntersection[predictedLabelsRes[k]] = accumIntersection[predictedLabelsRes[k]] + 1.0
                        accumUnion[predictedLabelsRes[k]] = accumUnion[predictedLabelsRes[k]] + 1.0
                    else:
                        accumUnion[labels[k]] = accumUnion[labels[k]] + 1.0
                        accumUnion[predictedLabelsRes[k]] = accumUnion[predictedLabelsRes[k]] + 1.0
                    accumGt[labels[k]] = accumGt[labels[k]] + 1.0

            accumLoss += lossRes

            accumTestLoss += lossRes

            #Compute Voxel accuracy
            resolution = 0.05
            coordMax = np.amax(points, axis=0)
            coordMin = np.amin(points, axis=0)
            nVoxels = np.ceil((coordMax-coordMin)/resolution)
            vidx = np.ceil((points-coordMin)/resolution)
            vidx = vidx[:,0]+vidx[:,1]*nVoxels[0]+vidx[:,2]*nVoxels[0]*nVoxels[1]
            uvidx = np.unique(vidx)
            voxelLabelCount = [np.bincount(labels[vidx==uv].astype(np.int32), minlength=len(semLabels)) for uv in uvidx]
            voxelPredLabelCount = [np.bincount(predictedLabelsRes[vidx==uv].astype(np.int32), minlength=len(semLabels)) for uv in uvidx]
            uvlabel = np.argmax(voxelLabelCount, axis = 1)
            uvpredlabel = np.argmax(voxelPredLabelCount, axis = 1)
            validVoxels = [1 if float(voxelLabelCount[k][0])/float(np.sum(voxelLabelCount[k])) < 0.3 and uvlabel[k] > 0 else 0 for k in range(len(uvidx))]

            for k in range(len(uvlabel)):
                if validVoxels[k] == 1:
                    if uvlabel[k] == uvpredlabel[k]:
                        accumVox[uvlabel[k]] = accumVox[uvlabel[k]] + 1.0
                    accumVoxGt[uvlabel[k]] = accumVoxGt[uvlabel[k]] + 1.0

            if (it+1)%10 == 0:
                visualize_progress(it, numTestRooms, ("Loss: %.6f") % (accumLoss/10.0))
                accumLoss = 0.0
            it += 1

        #Compute mean IoU
        print("############################## Category IoU / Acc / VoxAcc")
        meanIoUxCat = 0.0
        totalAccuracy = 0.0
        totalVoxAccuracy = 0.0
        totalIntersection = 0.0
        totalGt = 0.0
        for i in range(1, len(semLabels)):
            
            currMean = 0.0
            if accumUnion[i] <= 0.0:
                currMean = 1.0
            else:
                currMean = accumIntersection[i] / accumUnion[i]

            currAccuracy = 0.0
            if accumGt[i] <= 0.0:
                currAccuracy = 1.0
            else:
                currAccuracy = accumIntersection[i] / accumGt[i]

            currVoxAccuracy = 0.0
            if accumVoxGt[i] <= 0.0:
                currVoxAccuracy = 1.0
            else:
                currVoxAccuracy = accumVox[i] / accumVoxGt[i]

            totalIntersection = totalIntersection + accumIntersection[i]
            totalGt = totalGt + accumGt[i]

            print("Mean category "+semLabels[i]+": %.4f | %.4f | %.4f" % (currMean*100.0, currAccuracy*100.0, currVoxAccuracy*100.0))
            
            meanIoUxCat = meanIoUxCat + currMean
            totalAccuracy = totalAccuracy + currAccuracy
            totalVoxAccuracy = totalVoxAccuracy + currVoxAccuracy

        meanIoUxCat = meanIoUxCat / float(len(semLabels)-1)
        totalAccuracy = totalAccuracy / float(len(semLabels)-1)
        totalVoxAccuracy = totalVoxAccuracy / float(len(semLabels)-1)
        accumTestLoss = accumTestLoss/float(numTestRooms)
        noMeantotalAccuracy = totalIntersection / totalGt
        

        metricsTestSummRes = sess.run(metricsTestSummary, {iouVal: meanIoUxCat, accuracyVal : totalAccuracy, voxelAccuracyVal: totalVoxAccuracy})
        summary_writer.add_summary(metricsTestSummRes, step)

        #Print results
        print("############################## Global Accuracy and IoU")
        print("Loss: %.6f" % (accumTestLoss))
        print("Test total accuracy: %.4f [%.4f]" % (noMeantotalAccuracy*100.0, maxNoMeantotalAccuracy*100.0))
        print("Test accuracy: %.4f [%.4f]" % (totalAccuracy*100.0, maxAccuracy*100.0))
        print("Test voxel accuracy: %.4f [%.4f] " % (totalVoxAccuracy*100.0, maxVoxAccuracy*100.0))
        print("Test IoU %.4f [ %.4f ]" % (meanIoUxCat*100.0, maxIoU*100.0))
        with open(logFile, "a") as myfile:
            myfile.write("Loss: %.6f" % (accumTestLoss))
            myfile.write("Test total accuracy: %.4f [%.4f]" % (noMeantotalAccuracy*100.0, maxNoMeantotalAccuracy*100.0))
            myfile.write("Test accuracy: %.4f [%.4f]" % (totalAccuracy*100.0, maxAccuracy*100.0))
            myfile.write("Test voxel accuracy: %.4f [%.4f] " % (totalVoxAccuracy*100.0, maxVoxAccuracy*100.0))
            myfile.write("Test IoU %.4f [ %.4f ]" % (meanIoUxCat*100.0, maxIoU*100.0))
    
        saveModel = False
        if meanIoUxCat > maxIoU:
            maxIoU = meanIoUxCat
            saveModel = True
        if totalAccuracy > maxAccuracy:
            maxAccuracy = totalAccuracy
            saveModel = True
        if totalVoxAccuracy > maxVoxAccuracy:
            maxVoxAccuracy = totalVoxAccuracy
            saveModel = True
        if noMeantotalAccuracy > maxNoMeantotalAccuracy:
            maxNoMeantotalAccuracy = noMeantotalAccuracy
            saveModel = True

        if saveModel:
            saver.save(sess, args.logFolder+"/model.ckpt", global_step=epoch)

        endEpochTime = current_milli_time()   
        print("Epoch %3d  train and test time: %.4f" %(epoch, (endEpochTime-startEpochTime)/1000.0))
        with open(logFile, "a") as myfile:
            myfile.write("Epoch %3d  train and test time: %.4f \n" %(epoch, (endEpochTime-startEpochTime)/1000.0))


        #Increment epoch step variable for the learning rate decay
        sess.run(increment_epoch_step_op)
