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
import importlib
import os
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from PyUtils import visualize_progress, save_model
from ShapeNetDataSet import ShapeNetDataSet

current_milli_time = lambda: time.time() * 1000.0


def create_accuracy(logits, labels, scope):
    _, logitsIndexs = tf.nn.top_k(logits)
    with tf.variable_scope(scope):
        return tf.metrics.accuracy(labels, logitsIndexs)
                    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation of segmentation networks (ShapeNet)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--model', default='MCSeg', help='model (default: MCSeg)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    parser.add_argument('--nExec', default=1, type=int, help='Number of executions per model (default: 1)')
    parser.add_argument('--saveModels', action='store_true', help='Save models (default: False)')
    args = parser.parse_args()


    print("Trained model: "+args.inTrainedModel)
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("nExec: "+str(args.nExec))

    #Colors asigned to each part (used to save the model as a file).
    colors = [  [228,26,28],
                [55,126,184],
                [77,175,74],
                [152,78,163],
                [255,127,0],
                [255,255,51]]

    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    mTestDataSet = ShapeNetDataSet(False, args.nExec, 1.0, [0], False)
    cat = mTestDataSet.get_categories()
    segClasses = mTestDataSet.get_categories_seg_parts()
    print(segClasses)
    numTestModels = mTestDataSet.get_num_models()
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
    inLabels = tf.placeholder(tf.int32, [None, 1])
    isTraining = tf.placeholder(tf.bool)
    keepProbConv = tf.placeholder(tf.float32)
    keepProbFull = tf.placeholder(tf.float32)

    #Create the network
    logits = model.create_network(inPts, inBatchIds, inFeatures,inCatLabels, 1, len(cat), 50, 
        args.nExec, args.grow, isTraining, keepProbConv, keepProbFull, False, False)
          
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
    
    #Test the dataset.
    titleTexts = [
        "Uniform sampling",
        "Non-uniform split",
        "Non-uniform gradient",
        "Non-uniform lambert",
        "Non-uniform occlusion"]
    for samp in range(5):
        #Print the type of sampling used.
        print(titleTexts[samp])
        #Update the dataset.
        allowedSamplingsTest = [samp]
        mTestDataSet.set_allowed_samplings(allowedSamplingsTest)
        mTestDataSet.start_iteration()
        #Create the auxiliar variables.
        it = 0
        accumTime = 0.0
        step = 0
        epochStep = 0
        maxIoU = 0.0
        IoUxCat = [[] for i in range(len(cat))]
        #Iterate over the models.
        while mTestDataSet.has_more_batches():
            #Get the batch dataset.
            _, points, batchIds, features, labels, catLabels, modelsPath = mTestDataSet.get_next_batch(True)
        
            #Compute the predicted logits.
            startTimeMeasure = current_milli_time() 
            predictedLabelsRes, _ = sess.run([predictedLabels, accuracyAccumOp], 
                    {inPts: points, inBatchIds: batchIds, inFeatures: features, inCatLabels: catLabels, 
                    inLabels: labels, isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            endTimeMeasure = current_milli_time() 
            accumTime = accumTime + (endTimeMeasure - startTimeMeasure)
            
            #Save models
            if args.saveModels:
                save_model("savedModels/"+modelsPath[0].replace("/", "-")+"_sampling_"+
                    str(samp)+"_gt", points, labels, colors, 6)
                save_model("savedModels/"+modelsPath[0].replace("/", "-")+"_sampling_"+
                    str(samp)+"_pred", points, predictedLabelsRes.reshape((-1,1)), 
                    colors, 6)
            
            #Compute IoU
            numParts = len(segClasses[cat[catLabels[0][0]][0]])
            accumIoU = 0.0
            for j in range(numParts):
                intersection = 0.0
                union = 0.0
                currLabel = segClasses[cat[catLabels[0][0]][0]][j]
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
            IoUxCat[catLabels[0][0]].append(accumIoU)
            
            if it % 100 == 0:
                visualize_progress(it, numTestModels)
            
            it += 1

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

        totalAccuracy = sess.run(accuracyVal)

        print("Time: %.8f" % (accumTime/(float(numTestModels))))
        print("Test accuracy: %.4f | Test IoU %.4f [ %.4f ]" % (totalAccuracy*100.0, meanIoUxCat*100.0, maxIoU*100.0))