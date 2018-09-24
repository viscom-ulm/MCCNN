'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ModelNetDataSet.py

    \brief ModelNet dataset class.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import os
import math
import time
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from DataSet import DataSet

class ModelNetDataSet(DataSet):
    """ModelNet dataset.

    Attributes:
        useNormalsAsLabels_ (bool): Boolean that indicates if the normals will be used as the destination
                labels per each point. 
        useNormalsAsFeatures_ (bool): Boolean that indicates if the normals will be used as the input features.
        maxStoredNumPoints_ (int): Maximum number of points stored per model.
        catNames_ (string array): Name of the categories in the dataset.
    """
    
    def __init__(self, train, numPoints, ptDropOut, maxStoredNumPoints, batchSize, 
        allowedSamplings=[0], augment=False, useNormalsAsLabels=False, 
        useNormalsAsFeatures=False, folder="data", seed=None):
        """Constructor.

        Args:
            train (bool): Boolean that indicates if this is the train or test dataset.
            numPoints (int): Number of points that will be sampled from each model. If 0, all the 
                points of each model are used. 
            ptDropOut (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
            maxStoredNumPoints (int): Maximum number of points stored per model.
            batchSize (int): Size of the batch used.
            allowedSamplings (array of ints): Each element of the array determines an allowed sampling protocol
                that will be used to sample the different models. The implemented sampling protocols are:
                - 0: Uniform sampling
                - 1: Split sampling
                - 2: Gradient sampling
                - 3: Lambert sampling
                - 4: Occlusion sampling
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            useNormalsAsLabels (bool): Boolean that indicates if the normals will be used as the destination
                labels per each point.
            useNormalsAsFeatures (bool): Boolean that indicates if the normals will be used as the input features.
            folder (int): Folder in which the data is stored.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        
        # Store the parameters of the class.
        self.useNormalsAsLabels_ = useNormalsAsLabels
        self.useNormalsAsFeatures_ = useNormalsAsFeatures
        self.maxStoredNumPoints_ = maxStoredNumPoints

        # Create the list of labels that need to be augmented.
        augmentedLabels = []
        if useNormalsAsLabels:
            augmentedLabels = [0]

        # Create the list of features that need to be augmented.
        augmentedFeatures = []
        if useNormalsAsFeatures:
            augmentedFeatures = [0]

        # Call the constructor of the parent class.
        super(ModelNetDataSet,self).__init__(numPoints, ptDropOut, useNormalsAsFeatures, 
            useNormalsAsLabels, True, not(useNormalsAsLabels), False, batchSize, 
            allowedSamplings, 100000000, 0, augment, 1, True, True, augmentedFeatures, 
            augmentedLabels, seed)

        # Get the category names.
        self.catNames_ =[]
        with open(folder+"/modelnet40_shape_names.txt", 'r') as nameFile:
            for line in nameFile:
                self.catNames_.append(line.replace("\n",""))

        # List of files
        fileList = folder+"/modelnet40_test.txt"
        if train:
            fileList = folder+"/modelnet40_train.txt"
        with open(fileList, 'r') as nameFile:
            for line in nameFile:
                catId = -1
                for i in range(len(self.catNames_)):
                    if self.catNames_[i] in line:
                        catId = i
                        break
                if catId >= 0:
                    self.fileList_.append(folder+"/"+self.catNames_[catId]+"/"+line.replace("\n","")+".txt")
                    self.categories_.append(catId)
                    self.numPts_.append(self.maxStoredNumPoints_)
                    
    
    def get_categories(self):
        """Method to get the list of categories.
            
        Returns:
            pts (n np.array string): List of categories.
        """
        return self.catNames_


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
        
        fileDataArray = []
        with open(modelPath, 'r') as modelFile:        
            it = 0
            for line in modelFile:
                if it < self.maxStoredNumPoints_:
                    line = line.replace("\n", "")
                    currPoint = line.split(',')
                    fileDataArray.append([float(currPoint[0]), float(currPoint[1]), 
                        float(currPoint[2]), float(currPoint[3]), float(currPoint[4]), 
                        float(currPoint[5])])
                    it+=1
                else:
                    break
        fileData = np.array(fileDataArray)

        pts = fileData[:,0:3]
        normals = fileData[:,3:6]
        features = None
        if self.useNormalsAsFeatures_:
            features = normals
        labels = None
        if self.useNormalsAsLabels_:
            labels = normals
            
        return  pts, normals, features, labels