'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ShapeNet.py

    \brief Code to evaluate a segmentation network on the ScanNet dataset.

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

from PyUtils import visualize_progress, save_model
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to train MCCNN for segmentation tasks (S3DIS)')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--model', default='MCSegScanNet', help='model (default: MCSegScanNet)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--nExec', default=1, type=int, help='Number of executions per model (default: 1)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=1.0, type=float, help='GPU memory used (default: 1.0)')
    parser.add_argument('--useColor', action='store_true', help='Augment data (default: False)')
    parser.add_argument('--saveModels', action='store_true', help='Save models (default: False)')
    args = parser.parse_args()

    print("Trained Model: "+args.inTrainedModel)
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("Num executions: "+str(args.nExec))
    print("Use color: "+str(args.useColor))

    objColors = [   [0,0,0],        # Unannotated
                    [174,198,232],  # Wall
                    [151,223,137],  # Floor
                    [187,188,34],   # Chair
                    [254,151,150],  # Table
                    [247,183,210],  # Desk
                    [255,188,120],  # Bed
                    [148,103,188],  # Bookshelf
                    [140,86,74],    # Sofa
                    [112,128,144],  # Sink
                    [226,118,193],  # Bathtub
                    [42,159,44],    # Toilet
                    [218,219,141],  # Curtain
                    [23,190,208],   # Counter
                    [213,39,40],    # Door
                    [196,176,213],  # Window
                    [158,218,229],  # Shower curtain
                    [254,127,14],   # Refrigerator
                    [196,156,148],  # Picture
                    [31,120,180],   # Cabinet
                    [82,83,163]]    # Other furniture
    
    #Save models, create folder
    if args.saveModels:
        if not os.path.exists("savedModels"): os.mkdir("savedModels")

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    mTestDataSet = ScanNetDataSet(2, 1, 1.0, 0, False, args.useColor)
    semLabels = mTestDataSet.get_labels()
    print(semLabels)
    numTestRooms = mTestDataSet.get_num_models()
    print("Num test rooms: " + str(numTestRooms))
    
    #Create variable and place holders
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
    
    #Create the network
    numInputs = 1
    if args.useColor:
        numInputs =  4
    logits = model.create_network(inPts, inBatchIds, inFeatures, numInputs, len(semLabels), 1, 
        args.grow, isTraining, keepProbConv, keepProbFull, False, False)
          
    #Create predict labels
    predictedLabels = tf.argmax(logits, 1)
    
    #Create loss
    xentropyLoss, regularizationLoss = create_loss(logits, inLabels, inWeights, 0.0)
    loss = xentropyLoss + regularizationLoss

    #Create accuracy metric
    accuracyVal, accuracyAccumOp = create_accuracy(logits, inLabels, inAccWeights, 'metrics')
    metricsVars = tf.contrib.framework.get_variables('metrics', collection=tf.GraphKeys.LOCAL_VARIABLES)
    resetMetrics = tf.variables_initializer(metricsVars)

    #Create init variables 
    init = tf.global_variables_initializer()
    initLocal = tf.local_variables_initializer()

    #Create session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpuMem, visible_device_list=args.gpu)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    #create the saver
    saver = tf.train.Saver()   
    
    #Init variables
    sess.run(init)
    sess.run(initLocal)
    np.random.seed(int(time.time()))
    
    #Restore the model
    saver.restore(sess, args.inTrainedModel)
    
    #Test data
    print("############################## Evaluation" )
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

        _, points, batchIds, features, labels, _, sceneName = mTestDataSet.get_next_batch()
        currAccWeights = mTestDataSet.get_accuracy_masks(labels)

        for iterExec in range(args.nExec):
            
            lossRes, predictedLabelsRes = sess.run([loss, predictedLabels], 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inWeights: currAccWeights, 
                inAccWeights: currAccWeights, inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            

            #Save models
            if args.saveModels:
                save_model("savedModels/"+sceneName[0]+"_gt", points, labels, objColors)
                save_model("savedModels/"+sceneName[0]+"_pred", points, predictedLabelsRes.reshape((-1, 1)), 
                    objColors)
            
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
            resolution = 0.02
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

        visualize_progress(it, numTestRooms, ("Loss: %.6f "+sceneName[0]) % (accumLoss/(args.nExec)))
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
    accumTestLoss = accumTestLoss/float(numTestRooms*args.nExec)
    noMeantotalAccuracy = totalIntersection / totalGt

    #Print results
    print("############################## Global Accuracy and IoU")
    print("Loss: %.6f" % (accumTestLoss))
    print("Test total accuracy: %.4f" % (noMeantotalAccuracy*100.0))
    print("Test accuracy: %.4f" % (totalAccuracy*100.0))
    print("Test voxel accuracy: %.4f" % (totalVoxAccuracy*100.0))
    print("Test IoU %.4f" % (meanIoUxCat*100.0))
