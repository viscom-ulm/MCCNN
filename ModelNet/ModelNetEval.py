'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ModelNetEval.py

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
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from PyUtils import visualize_progress
from ModelNetDataSet import ModelNetDataSet

current_milli_time = lambda: time.time() * 1000.0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation of classification networks (ModelNet40)')
    parser.add_argument('--inTrainedModel', default='log/model.ckpt', help='Input trained model (default: log/model.ckpt)')
    parser.add_argument('--model', default='MCClass', help='model (default: MCClass)')
    parser.add_argument('--grow', default=32, type=int, help='Grow rate (default: 32)')
    parser.add_argument('--nPoints', default=1024, type=int, help='Number of points (default: 1024)')
    parser.add_argument('--nExec', default=1, type=int, help='Number of executions per model (default: 1)')
    parser.add_argument('--gpu', default='0', help='GPU (default: 0)')
    parser.add_argument('--gpuMem', default=0.5, type=float, help='GPU memory used (default: 0.5)')
    args = parser.parse_args()

    print("Trained model: "+args.inTrainedModel)
    print("Model: "+args.model)
    print("Grow: "+str(args.grow))
    print("nPoints: "+str(args.nPoints))
    print("nExec: "+str(args.nExec))
    
    #Load the model
    model = importlib.import_module(args.model)

    #Get train and test files
    mTestDataSet = ModelNetDataSet(False, args.nPoints, 1.0, 5000, 
        args.nExec, [0], False)
    numTestModels = mTestDataSet.get_num_models()
    categories = mTestDataSet.get_categories()
    print(categories)
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
    logits = model.create_network(inPts, inBatchIds, inFeatures, 1, args.nExec, args.grow, 
        len(categories), isTraining, keepProbFull, keepProbConv, False, False)
          
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
        i = 0
        accumTime = 0.0
        totalAccuracy = 0.0
        #Iterate over the models.
        while mTestDataSet.has_more_batches():
            #Get the batch dataset.
            _, points, batchIds, features, _, labels, _ = mTestDataSet.get_next_batch(True)
        
            #Compute the predicted logits.
            startTimeMeasure = current_milli_time() 
            logitsRes = sess.run(logits, 
                {inPts: points, inBatchIds: batchIds, inFeatures: features, inLabels: labels, 
                isTraining: False, keepProbConv: 1.0, keepProbFull: 1.0})
            endTimeMeasure = current_milli_time() 
            accumTime = accumTime + (endTimeMeasure - startTimeMeasure)
            
            #Print the progress.
            if i%100 == 0:
                visualize_progress(i, numTestModels)
        
            #Compute the predicted class.
            predCat = np.argmax(np.sum(logitsRes, axis=0))
            if predCat == labels[0]:
                totalAccuracy = totalAccuracy + 1.0
                
            i += 1
            
        #Print the results.
        print("Time: %.8f" % (accumTime/(float(numTestModels))))
        totalAccuracy = totalAccuracy/float(numTestModels)
        print("Test accuracy: %.4f" % (totalAccuracy*100.0))
    