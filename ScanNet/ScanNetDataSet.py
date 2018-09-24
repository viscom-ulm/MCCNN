'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ShapeNetDataSet.py

    \brief ScanNet dataset class.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import time
import json
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from DataSet import DataSet

class ScanNetDataSet(DataSet):
    """ScanNet dataset.

    Attributes: 
        useColorsAsFeatures_ (bool): Boolean that indicates if the colors will be used as the input features.
        dataFolder_ (string): Path of the folder with the data.
        weights_ (array of floats): List of weights for each label or category.
    """
    
    def __init__(self, dataset, batchSize, ptDropOut, maxNumPtsxBatch=600000,
        augment=False, useColorsAsFeatures=False, dataFolder="data_mccnn", seed=None):
        """Constructor.

        Args:
            dataset (int): Index of the dataset that will be used. 0 - train, 1 - val, 2 - test
            batchSize (int): Size of the batch used.
            ptDropOut (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            useColorsAsFeatures (bool): Boolean that indicates if the colors will be used as the input features.
            dataFolder (string): Path of the folder with the data.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        
        # Check if all the models will fit in the batch.
        if (maxNumPtsxBatch < 600000) and (maxNumPtsxBatch > 0):
            raise RuntimeError('The number of points per batch should be big enough to store'\
                ' all the models (greater than 600000).')
            
        # Store the parameters of the class.
        self.useColorsAsFeatures_ = useColorsAsFeatures
        self.dataFolder_ = dataFolder

        # Call the constructor of the parent class.
        super(ScanNetDataSet,self).__init__(0, ptDropOut, 
            useColorsAsFeatures, True, False,
            False, False, batchSize, [0], 0, maxNumPtsxBatch,
            augment, 2, False, False, [], [], seed)

        # Load the room list.
        rooms = np.loadtxt(dataFolder+"/rooms.txt", dtype='str')
        # Load the number of points per rooms.
        roomsNummPoints = np.loadtxt(dataFolder+"/num_points.txt")*ptDropOut
        # Room per dataset.
        trainRooms = set(np.loadtxt(dataFolder+"/scannet_train.txt", dtype='str'))
        valRooms = set(np.loadtxt(dataFolder+"/scannet_val.txt", dtype='str'))
        testRooms = set(np.loadtxt(dataFolder+"/scannet_test.txt", dtype='str'))
        # Determine the indexs of the rooms for each dataset.
        roomIndexs = np.array([])
        if dataset == 0:
            roomIndexs = np.array([i for i in range(len(rooms)) if rooms[i] in trainRooms])
        elif dataset == 1:
            roomIndexs = np.array([i for i in range(len(rooms)) if rooms[i] in valRooms])
        else:
            roomIndexs = np.array([i for i in range(len(rooms)) if rooms[i] in testRooms])

        # Store the file rooms of the dataset with the number of points.
        self.fileList_ = rooms[roomIndexs]#["scene0497_00"]
        self.numPts_ = roomsNummPoints[roomIndexs]#[590000]

        # Load the labels identifiers and the weights.
        self.semLabels_ = np.loadtxt(dataFolder+"/labels.txt", dtype='str', delimiter=':')
        weights = np.loadtxt(dataFolder+"/weights.txt")
        for i in range(len(self.semLabels_)):
            weights[0][i] = 1.0/np.log(1.2 + weights[0][i])
            weights[1][i] = 1.0/np.log(1.2 + weights[1][i])
        self.weights_ = weights[0]
        self.weights_[0] = 0.0

        
    def get_labels(self):
        """Method to get the list of labels.
            
        Returns:
            pts (n np.array string): List of labels.
        """
        return self.semLabels_

        
    def get_weights(self, labels):
        """Method to get the weights associated to the labels.
            
        Args:
            catlabs (nxm np.array): Labels for which we want the weights.
        Returns:
            weights (nxm): Weights associated with the input labels.
        """
        
        if len(self.weights_) == 0:
            raise RuntimeError('No weights associated to the labels.')
            
        outWeights = np.array([[self.weights_[currLab[0]]] for currLab in labels])
        return outWeights
        
    
    def get_accuracy_masks(self, labels):
        """Method to get the list of mask for each label to compute the accuracy.
            
        Args:
            labels (np.array): Labels for which we want the weights.
        Returns:
            masks (np.array): List of mask for each label to compute 
                the accuracy.
        """
        outMasks = np.array([[1.0] if lab[0] != 0 else [0.0] for lab in labels])
        return outMasks


    def _load_model_from_disk_(self, modelPath):
        """Abstract method that should be implemented by child class which loads a model
            from disk.

        Args:
            modelPath (string): Path to the model that needs to be loaded.

        Returns:
            pts (nx3 np.array): List of points.
            normals (nx3 np.array): List of normals. If the dataset does not contain 
                normals, None should be returned.
            features (nxm np.array): List of features. If the dataset does not contain
                features, None should be returned.
            labels (nxl np.array): List of labels. If the dataset does not contain
                labels, None should be returned.
        """
        
        pts = np.load(self.dataFolder_+"/"+modelPath+"_pos.npy")
        labels = np.load(self.dataFolder_+"/"+modelPath+"_labels.npy")        
        normals = None
        features = None
        if self.useColorsAsFeatures_:
            features = np.load(self.dataFolder_+"/"+modelPath+"_colors.npy")
        centroid = np.mean(pts, axis= 0)
        pts = pts - centroid
            
        return  pts, normals, features, labels.reshape((-1,1))