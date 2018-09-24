'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file ShapeNetDataSet.py

    \brief ShapeNet dataset class.

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

class ShapeNetDataSet(DataSet):
    """ShapeNet dataset.

    Attributes: 
        useNormalsAsFeatures_ (bool): Boolean that indicates if the normals will be used as the input features.
        cat_ (nx2 array): List of tuples (category name, category folder) of the categories in the dataset.
        segClasses_ (dictionary of arrays): Each entry of the dictionary has a key equal to the name of the
                category and a list of part identifiers.
    """
    
    def __init__(self, train, batchSize, ptDropOut, allowedSamplings=[0], augment=False, 
        useNormalsAsFeatures=False, seed=None):
        """Constructor.

        Args:
            train (bool): Boolean that indicates if this is the train or test dataset.
            batchSize (int): Size of the batch used.
            ptDropOut (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
            allowedSamplings (array of ints): Each element of the array determines an allowed sampling protocol
                that will be used to sample the different models. The implemented sampling protocols are:
                - 0: Uniform sampling
                - 1: Split sampling
                - 2: Gradient sampling
                - 3: Lambert sampling
                - 4: Occlusion sampling
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            useNormalsAsFeatures (bool): Boolean that indicates if the normals will be used as the input features.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """
        
        # Store the parameters of the class.
        self.useNormalsAsFeatures_ = useNormalsAsFeatures

        # Create the list of features that need to be augmented.
        augmentedFeatures = []
        if useNormalsAsFeatures:
            augmentedFeatures = [0]

        # Call the constructor of the parent class.
        super(ShapeNetDataSet,self).__init__(0, ptDropOut, useNormalsAsFeatures, True, True,
            True, True, batchSize, allowedSamplings, 100000000, 0,
            augment, 1, True, False, augmentedFeatures, [], seed)

        # Get the categories and their associated part ids..
        self.catNames_ = []
        with open("./shape_data/synsetoffset2category.txt", 'r') as nameFile:
            for line in nameFile:
                strings = line.replace("\n", "").split("\t")
                self.catNames_.append((strings[0], strings[1]))

        self.segClasses_ = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 
            'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 
            'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # List of files.
        if train:
            with open("./shape_data/train_test_split/shuffled_train_file_list.json", 'r') as f:
                self.fileList_ = list([d for d in json.load(f)])
            with open("./shape_data/train_test_split/shuffled_val_file_list.json", 'r') as f:
                self.fileList_ = self.fileList_ + list([d for d in json.load(f)])
        else:
            with open("./shape_data/train_test_split/shuffled_test_file_list.json", 'r') as f:
                self.fileList_ = list([d for d in json.load(f)])

        # Check the categories per model.
        for currModel in self.fileList_:
            catId = 0
            for currCat in range(len(self.catNames_)):
                if self.catNames_[currCat][1] in currModel:
                    catId = currCat
            self.categories_.append(catId)

        # Since we do not know the size of the models in advance we initialize them to 0 and the first that will be loaded
        # this values will be update automatically.
        self.numPts_ = [0 for i in range(len(self.fileList_))] 

    
    def get_categories(self):
        """Method to get the list of categories.
            
        Returns:
            pts (nx2 np.array string): List of tuples with the category name and the folder name.
        """
        return self.catNames_


    def get_categories_seg_parts(self):
        """Method to get the list of parts per category.
            
        Returns:
            pts (dict of array): Each entry of the dictionary has a key equal to the name of the
                category and a list of part identifiers.
        """
        return self.segClasses_


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
        with open(modelPath+".txt", 'r') as modelFile:        
            for line in modelFile:
                line = line.replace("\n", "")
                currPoint = line.split()
                fileDataArray.append([float(currPoint[0]), float(currPoint[1]), 
                    float(currPoint[2]), float(currPoint[3]), float(currPoint[4]), 
                    float(currPoint[5]), float(currPoint[6])])
        fileData = np.array(fileDataArray)

        pts = fileData[:,0:3]
        normals = fileData[:,3:6]
        features = None
        if self.useNormalsAsFeatures_:
            features = normals
        labels = fileData[:,6:7]
            
        return  pts, normals, features, labels