'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file DataSet.py

    \brief Base class for a dataset.

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
import abc 
from six import with_metaclass
from collections import deque

class DataSet(with_metaclass(abc.ABCMeta)):
    """Base class of a dataset.

    Attributes:
        numPoints_ (int): Number of points that will be sampled from each model. If 0, all the 
            points of each model are used. 
        ptDropOut_ (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
        pointFeatures_ (bool): Boolean that indicates if the models contain point features. If not,
            only one feature will be generated for each point with value equal to 1.0. 
        pointLabels_ (bool): Boolean that indicates if the models contain point labels. 
        pointNormals_ (bool): Boolean that indicates if the models contain point normals. If not,
            sampling protocols lambert and occlusion cannot be used.
        useCategories_ (bool): Boolean that indicates if there dataset contains categories for each model.
            If True, the class will generate a list with an integer per each model in the batch.
        pointCategories_ (bool): Boolean that indicates if the category of the model should be provided
            for each point indepently. If True, instead of providing a list with an integer per each
            model in the dataset, a list with the category of the model for each point will be generated.
        batchSize_ (int): Size of the batch used.
        allowedSamplings_ (array of ints): Each element of the array determines an allowed sampling protocol
            that will be used to sample the different models. The implemented sampling protocols are:
            - 0: Uniform sampling
            - 1: Split sampling
            - 2: Gradient sampling
            - 3: Lambert sampling
            - 4: Occlusion sampling
        maxNumPtsCache_ (int): Maximum number of points that can be stored in the cache.
        maxPtsxBatch_ (int): Maximum number of points per batch. If 0, each batch will contain batchSize 
            number of models. If greater than 0, the batch will contain a maximum number of models equal 
            to batchSize or a maximum number of point equal to maxPtsxBatch.
        augment_ (bool): Boolean that indicates if data augmentation will be used in the models.
        augmentMainAxis_ (int): Indexs of the axis used to rotate the models when data augmentation is 
            active. Possible values (0, 1, 2).
        augmentSmallRotations_ (bool): Boolean that indicates if small rotations alogn every axis will be used
            when data augmentation is active.
        uniformSelectFirst_ (bool): Boolean that indicates if the first n points will be selected from the list
            in the uniform sampling protocol, or a random selection of n points will be used instead.
        augmentedFeatures_ (array of ints): Each element of the list points to the first element of groups of
            3 features that will also processed by the data augmentation. Example: the value 2 in the list will 
            indicate that features f[2:5] will be treated as a vector an rotated by the data augmentation algorithm.
            This feature will allow to use data augmentation when vectors are provided as features to the networks
            (normals, tangent space, velocity, etc.).
        augmentedLabels_ (array of ints): The same as augmentedFeatures_, but for labels.
        fileList_ (array of strings): List of paths of the models of the dataset.
        numPts_ (array of ints): List of number of points of each model of the dataset.
        categories_ (array of ints): List of categories of each model of the dataset. If the dataset does not contain
            category information, the list is empty.
        cacheCurrNumPts (int): Number of points stored in the cache.
        cacheAddedOrder_ (deque of strings): List of last accessed models. The first element of the queue was the last 
            queried model.
        ptsCache_ (dict of arrays): Cache of point lists.
        normalsCache_ (dict of arrays): Cache of normals lists.
        featureCache_ (dict of arrays): Cache of feature lists.
        labelsCache_ (dict of arrays): Cache of labels lists.
        randomSelection_ (array of ints): Permutated indexs of the models of the dataset.
        iterator_ (int): Iterator pointing at the next model to process in the randomSelection_ list.
        randomState_ (random state): Numpy random state used to generate random numbers.
    """

    def __init__(self, numPoints, ptDropOut, pointFeatures, pointLabels, 
        pointNormals, useCategories, pointCategories, batchSize, allowedSamplings,
        maxNumPtsCache=100000000, maxPtsxBatch=0, augment=False, augmentMainAxis=1, 
        augmentSmallRotations=False, uniformSelectFirst=False, augmentedFeatures=[], 
        augmentedLabels=[], seed=None):
        """Constructor.

        Args:
            numPoints (int): Number of points that will be sampled from each model. If 0, all the 
                points of each model are used. 
            ptDropOut (float): Probability to keep a point during uniform sampling when all the points
                or only the first n number of points are selected.
            pointFeatures (bool): Boolean that indicates if the models contain point features. If not,
                only one feature will be generated for each point with value equal to 1.0. 
            pointLabels (bool): Boolean that indicates if the models contain point labels. 
            pointNormals (bool): Boolean that indicates if the models contain point normals. If not,
                sampling protocols lambert and occlusion cannot be used.
            useCategories (bool): Boolean that indicates if there dataset contains categories for each model.
                If True, the class will generate a list with an integer per each model in the batch.
            pointCategories (bool): Boolean that indicates if the category of the model should be provided
                for each point indepently. If True, instead of providing a list with an integer per each
                model in the dataset, a list with the category of the model for each point will be generated.
            batchSize (int): Size of the batch used.
            allowedSamplings (array of ints): Each element of the array determines an allowed sampling protocol
                that will be used to sample the different models. The implemented sampling protocols are:
                - 0: Uniform sampling
                - 1: Split sampling
                - 2: Gradient sampling
                - 3: Lambert sampling
                - 4: Occlusion sampling
            maxNumPtsCache (int): Maximum number of points that can be stored in the cache.
            maxPtsxBatch (int): Maximum number of points per batch. If 0, each batch will contain batchSize 
                number of models. If greater than 0, the batch will contain a maximum number of models equal 
                to batchSize or a maximum number of point equal to maxPtsxBatch.
            augment (bool): Boolean that indicates if data augmentation will be used in the models.
            augmentMainAxis (int): Indexs of the axis used to rotate the models when data augmentation is 
                active. Possible values (0, 1, 2).
            augmentSmallRotations (bool): Boolean that indicates if small rotations alogn every axis will be used
                when data augmentation is active.
            uniformSelectFirst (bool): Boolean that indicates if the first n points will be selected from the list
                in the uniform sampling protocol, or a random selection of n points will be used instead.
            augmentedFeatures (array of ints): Each element of the list points to the first element of groups of
                3 features that will also processed by the data augmentation. Example: the value 2 in the list will 
                indicate that features f[2:5] will be treated as a vector an rotated by the data augmentation algorithm.
                This feature will allow to use data augmentation when vectors are provided as features to the networks
                (normals, tangent space, velocity, etc.).
            augmentedLabels (array of ints): The same as augmentedFeatures_, but for labels.
            seed (int): Seed used to initialize the random number generator. If None is provided instead, the current
                time on the machine will be used to initialize the number generator.
        """

        # Check the allowed sampling protocols are valid.
        if not((augmentMainAxis >=0) and (augmentMainAxis<3)):
            raise RuntimeError('Invalid augmentMainAxis')

        # Check the allowed sampling protocols are valid.
        correctSamplings = True
        for sampl in allowedSamplings:
            correctSamplings = correctSamplings and ((sampl >=0) and (sampl < 5))
        if not correctSamplings:
            raise RuntimeError('Invalid sampling protocol')

        # Check that the dataset contains normals when it has to be used with the sampling protocols
        # lambert and occlusion.
        if ((3 in list(allowedSamplings)) or (4 in list(allowedSamplings))) and (not(pointNormals)):
            raise RuntimeError('The dataset should contain normals in order to use the sampling protocols '\
                'lambert and occlusion')

        # Check that the groups of 3 features that need to be augmented by rotation do not overlap.
        correct = True
        for i in range(max(len(augmentedFeatures)-1,0)):
            correct = correct and ((augmentedFeatures[i+1]-augmentedFeatures[i]) >= 3)
        for i in range(max(len(augmentedLabels)-1,0)):
            correct = correct and ((augmentedLabels[i+1]-augmentedLabels[i]) >= 3)
        if not correct:
            raise RuntimeError('The groups of 3 features/labels to augment should not overlap ')

        # Store the configuration parameters of the dataset.
        self.numPoints_ = numPoints
        self.ptDropOut_ = ptDropOut
        self.pointFeatures_ = pointFeatures
        self.pointLabels_ = pointLabels
        self.pointNormals_ = pointNormals
        self.useCategories_ = useCategories
        self.pointCategories_ = pointCategories
        self.batchSize_ = batchSize
        self.allowedSamplings_ = allowedSamplings
        self.maxNumPtsCache_ = maxNumPtsCache
        self.maxPtsxBatch_ = maxPtsxBatch
        self.augment_ = augment
        self.augmentMainAxis_ = augmentMainAxis
        self.augmentSmallRotations_ = augmentSmallRotations
        self.augmentedFeatures_ = augmentedFeatures
        self.augmentedLabels_ = augmentedLabels
        self.uniformSelectFirst_ = uniformSelectFirst
        
        # Prepare the empty lists for the models of the dataset.
        self.fileList_ = []
        self.numPts_ = []
        self.categories_ = []
        self.weights_ = []

        # Initialize the cache.
        self.cacheCurrNumPts = 0
        self.cacheAddedOrder_ = deque([])
        self.cache_ = {}

        # Initialize the lists used to iterate over the models of the dataset.
        self.randomSelection_ = []
        self.iterator_ = 0

        # Initialize the random seed.
        if not(seed is None):
            self.randomState_ = np.random.RandomState(seed)
        else:
            self.randomState_ = np.random.RandomState(int(time.time()))


    @abc.abstractmethod
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
        pass

    def _load_model_(self, modelPath):
        """Method to load a model from disk or cache. If the model is loaded from distk,
        the cache is updated.

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

        # Load the current model from disk or the cache.
        if modelPath in self.cache_:
            # Cache
            currTuple = self.cache_[modelPath]
            pts = currTuple[0]
            normals = currTuple[1]
            features = currTuple[2]
            labels = currTuple[3]
            self.cacheAddedOrder_.remove(modelPath)
            self.cacheAddedOrder_.append(modelPath)
        else:
            # Load the model from disk.
            pts, normals, features, labels = self._load_model_from_disk_(modelPath)

            # Check if the cache is active.
            if self.maxNumPtsCache_ > 0:
                # Is the cache full?
                if (self.cacheCurrNumPts+len(pts)) > self.maxNumPtsCache_:
                    modelToRemove = self.cacheAddedOrder_.popleft()
                    del self.cache_[modelToRemove]
                    numPtsToRemove = self.fileList_.index(modelToRemove)
                    self.cacheCurrNumPts -= numPtsToRemove

                # Add the model to the cache.
                self.cache_[modelPath] = (pts, normals, features, labels)
                self.cacheAddedOrder_.append(modelPath)
                self.cacheCurrNumPts += len(pts)
        
        return pts, normals, features, labels

    def _augment_data_rot_(self, inData, mainRotAxis = 1, smallRotations = False, inRotationMatrix = None):
        """Method to augment a list of vectors by rotating alogn an axis, and perform
        small rotations alogn all the 3 axes.

        Args:
            inData (nx3 np.array): List of vectors to augment. 
            mainRotAxis (int): Rotation axis. Allowed values (0, 1, 2).
            smallRotations (bool): Boolean that indicates if small rotations alogn all the
                3 axes will be also applied. 
            inRotationMatrix (3x3 np.array): Transformation matrix. If provided, no matrix is computed 
                and this parameter is used instead for the transformations.

        Returns:
            augData (nx3 np.array): List of transformed vectors.
            rotation_matrix (3x3 np.array): Transformation matrix used to augment the data.
        """
        
        rotationMatrix = inRotationMatrix
        if inRotationMatrix is None:
            # Compute the main rotation
            rotationAngle = self.randomState_.uniform() * 2.0 * np.pi
            cosval = np.cos(rotationAngle)
            sinval = np.sin(rotationAngle)
            if mainRotAxis == 0:
                rotationMatrix = np.array([[1.0, 0.0, 0.0], [0.0, cosval, -sinval], [0.0, sinval, cosval]])
            elif mainRotAxis == 1:
                rotationMatrix = np.array([[cosval, 0.0, sinval], [0.0, 1.0, 0.0], [-sinval, 0.0, cosval]])
            else:
                rotationMatrix = np.array([[cosval, -sinval, 0.0], [sinval, cosval, 0.0], [0.0, 0.0, 1.0]])
            
            # Compute small rotations.
            if smallRotations:
                angles = np.clip(0.06*self.randomState_.randn(3), -0.18, 0.18)
                Rx = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(angles[0]), -np.sin(angles[0])],
                            [0.0, np.sin(angles[0]), np.cos(angles[0])]])
                Ry = np.array([[np.cos(angles[1]), 0.0, np.sin(angles[1])],
                            [0.0, 1.0, 0.0],
                            [-np.sin(angles[1]), 0.0, np.cos(angles[1])]])
                Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0.0],
                            [np.sin(angles[2]), np.cos(angles[2]), 0.0],
                            [0.0, 0.0, 1.0]])
                R = np.dot(Rz, np.dot(Ry,Rx))
                rotationMatrix = np.dot(R, rotationMatrix)

        # Transform data
        return np.dot(inData[:,0:3].reshape((-1, 3)), rotationMatrix), rotationMatrix


    def _uniform_sampling_(self, points, inNumPoints, selectFirst, inFeatures=None, inLabels=None, numPoints=0):
        """Method to uniformly sample a point cloud.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list. 
            selectFirst (bool): Boolean that indicates if the first numPoints should be selected or a random
                selection should be used instead.
            inFeatures (nxm np.array): List of point features. 
            inLabels (nxl np.array): List of point labels. 
            numPoints (int): Number of points to sample. If 0, all the points are selected.

        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """
        
        choice = np.array([])
        if numPoints > 0:
            # If the caller asked for the first numPoints points, return the input
            # after applying the point dropout
            if selectFirst and (inNumPoints >= numPoints):
                choice = self.randomState_.choice(int(float(numPoints)*(2.0-self.ptDropOut_)), 
                    numPoints, replace=False)
            
            # We select a random number of points. If the input number of points is smaller
            # than the number of points requested, we allow for repetition on the selected
            # points.
            else:
                choice = self.randomState_.choice(inNumPoints, numPoints, 
                    replace=(inNumPoints < numPoints))

        else:
            # If the caller did not ask for a specific number of points, return the input
            # after applying the point dropout.
            choice = self.randomState_.choice(inNumPoints, int(float(inNumPoints)*self.ptDropOut_), 
                replace=False)
        
        # Return the result.
        auxOutPts = points[choice]
        auxOutInFeatures = None
        if not(inFeatures is None):
            auxOutInFeatures =  inFeatures[choice]
        auxOutInLabels = None
        if not(inLabels is None):
            auxOutInLabels = inLabels[choice]
        return auxOutPts, auxOutInFeatures, auxOutInLabels


    def _non_uniform_sampling_split_(self, points, inNumPoints, inFeatures=None, inLabels=None, numPoints=0, lowProbability=0.25):
        """Method to non-uniformly sample a point cloud using the split protocol. Point of half of the bounding box are 
        selected with a probability of 1.0 and the other half with a probability of lowProbability.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list. 
            inFeatures (nxm np.array): List of point features. 
            inLabels (nxl np.array): List of point labels. 
            numPoints (int): Number of points to sample. If 0, all the points are selected.
            lowProbability (float): Probability used for the points in the second half of the bounding box.
        
        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """

        # Compute the bounding box.
        coordMax = np.amax(points, axis=0)
        coordMin = np.amin(points, axis=0)
        aabbSize = coordMax - coordMin
        largestAxis = np.argmax(aabbSize)

        auxOutPts = []
        auxOutInFeatures = []
        auxOutInLabels = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Compute the point probability.
                currPt = points[i]
                ptPos = (currPt[largestAxis]-coordMin[largestAxis])/aabbSize[largestAxis]
                if ptPos > 0.5:
                    probVal = 1.0
                else:
                    probVal = lowProbability
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(currPt)
                    if not(inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    if not(inLabels is None):
                        auxOutInLabels.append(inLabels[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        npOutInFeatures = None
        if not(inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
        npOutInLabels = None
        if not(inLabels is None):
            npOutInLabels = np.array(auxOutInLabels)
        return npOutPts, npOutInFeatures, npOutInLabels


    def _non_uniform_sampling_gradient_(self, points, inNumPoints, inFeatures=None, inLabels=None, numPoints=0):
        """Method to non-uniformly sample a point cloud using the gradient protocol. The probability to select a
        point is based on its position alogn the largest axis of the bounding box.

        Args:
            points (nx3 np.array): List of points.
            inNumPoints (int): Number of points in the list. 
            inFeatures (nxm np.array): List of point features. 
            inLabels (nxl np.array): List of point labels. 
            numPoints (int): Number of points to sample. If 0, all the points are selected.

        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """
        # Compute the bounding box.
        coordMax = np.amax(points, axis=0)
        coordMin = np.amin(points, axis=0)
        aabbSize = coordMax - coordMin
        largestAxis = np.argmax(aabbSize)

        auxOutPts = []
        auxOutInFeatures = []
        auxOutInLabels = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Compute the point probability.
                currPt = points[i]
                probVal = (currPt[largestAxis]-coordMin[largestAxis]-aabbSize[largestAxis]*0.2
                    )/(aabbSize[largestAxis]*0.6)
                probVal = pow(np.clip(probVal, 0.01, 1.0), 1.0/2.0)
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(currPt)
                    if not(inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    if not(inLabels is None):
                        auxOutInLabels.append(inLabels[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        npOutInFeatures = None
        if not(inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
        npOutInLabels = None
        if not(inLabels is None):
            npOutInLabels = np.array(auxOutInLabels)
        return npOutPts, npOutInFeatures, npOutInLabels

    
    def _non_uniform_sampling_lambert_(self, viewDir, points, normals, inNumPoints, inFeatures=None, 
        inLabels=None, numPoints=0):
        """Method to non-uniformly sample a point cloud using the lambert protocol. The probability to select a
        point is based on the dot product between the point normal and a view direction.

        Args:
            viewDir (3 np.array): View vector used to compute the probability of each point.
            points (nx3 np.array): List of points.
            normals (nx3 np.array): List of point normals.
            inNumPoints (int): Number of points in the list. 
            inFeatures (nxm np.array): List of point features. 
            inLabels (nxl np.array): List of point labels. 
            numPoints (int): Number of points to sample. If 0, all the points are selected.

        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """
        auxOutPts = []
        auxOutInFeatures = []
        auxOutInLabels = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Compute the point probability.
                probVal = np.dot(viewDir, normals[i])
                probVal = pow(np.clip(probVal, 0.0, 1.0), 0.5)
                # Determine if we select the point.
                rndNum = self.randomState_.random_sample()
                if rndNum < probVal:
                    # Store the point in the output buffers.
                    auxOutPts.append(points[i])
                    if not(inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    if not(inLabels is None):
                        auxOutInLabels.append(inLabels[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        npOutInFeatures = None
        if not(inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
        npOutInLabels = None
        if not(inLabels is None):
            npOutInLabels = np.array(auxOutInLabels)
        return npOutPts, npOutInFeatures, npOutInLabels


    def _non_uniform_sampling_occlusion_(self, viewDir, points, normals, inNumPoints, inFeatures=None, 
        inLabels=None, numPoints=0, screenResolution=128):
        """Method to non-uniformly sample a point cloud using the occlusion protocol. A point is selected
        if it is visible from the view direction.

        Args:
            viewDir (3 np.array): View vector used to compute the visibility of each point.
            points (nx3 np.array): List of points.
            normals (nx3 np.array): List of point normals.
            inNumPoints (int): Number of points in the list. 
            inFeatures (nxm np.array): List of point features. 
            inLabels (nxl np.array): List of point labels. 
            numPoints (int): Number of points to sample. If 0, all the points are selected.

        Returns:
            sampledPts (nx3 np.array): List of sampled points.
            sampledFeatures (nxm np.array): List of the features of the sampled points.
            sampledLabels (nxl np.array): List of the labels of the sampled points.
        """

        # Compute the screen plane.
        xVec = np.cross(viewDir, np.array([0.0, 1.0, 0.0]))
        xVec = xVec / np.linalg.norm(xVec)
        yVec = np.cross(xVec, viewDir)
        yVec = yVec / np.linalg.norm(yVec)
        
        # Compute the bounding box.
        coordMax = np.amax(points, axis=0)
        coordMin = np.amin(points, axis=0)
        diagonal = np.linalg.norm(coordMax - coordMin)*0.5
        center = (coordMax + coordMin)*0.5
        
        # Create the screen pixels
        screenSize = screenResolution
        pixelSize = diagonal/(float(screenSize)*0.5)
        screenPos = center - viewDir*diagonal - xVec*diagonal - yVec*diagonal
        screenZBuff = np.full([screenSize, screenSize], -1.0)
        
        # Compute the z value and pixel id in which each point is projected.
        pixelIds = [[-1, -1] for i in range(inNumPoints)]
        zVals = [1.0 for i in range(inNumPoints)]
        for i in range(inNumPoints):
            # If the point is facing the camera.
            if np.dot(normals[i], viewDir) < 0.0:
                # Compute the z value of the pixel.
                transPt = points[i] - screenPos
                transPt = np.array([
                    np.dot(transPt, xVec),
                    np.dot(transPt, yVec),
                    np.dot(transPt, viewDir)/(diagonal*2.0)])
                zVals[i] = transPt[2]

                # Compute the pixel id in which the point is projected.
                transPt = transPt / pixelSize
                pixelIds[i][0] = int(np.floor(transPt[0]))
                pixelIds[i][1] = int(np.floor(transPt[1]))
                    
                # Update the z-buffer.
                if screenZBuff[pixelIds[i][0]][pixelIds[i][1]] > zVals[i] or screenZBuff[pixelIds[i][0]][pixelIds[i][1]] < 0.0:
                    screenZBuff[pixelIds[i][0]][pixelIds[i][1]] = zVals[i]

        auxOutPts = []
        auxOutInFeatures = []
        auxOutInLabels = []
        exitVar = False
        numAddedPts = 0
        # Iterate over the points until we have the desired number of output points.
        while not exitVar:
            # Iterate over the points.
            for i in range(inNumPoints):
                # Determine if the point is occluded.
                if (zVals[i] - screenZBuff[pixelIds[i][0]][pixelIds[i][1]]) < 0.01:
                    # Store the point in the output buffers.
                    auxOutPts.append(points[i])
                    if not(inFeatures is None):
                        auxOutInFeatures.append(inFeatures[i])
                    if not(inLabels is None):
                        auxOutInLabels.append(inLabels[i])
                    numAddedPts += 1
                    if (numPoints > 0) and (numAddedPts >= numPoints):
                        exitVar = True
                        break

            if numPoints == 0:
                exitVar = True

        npOutPts = np.array(auxOutPts)
        npOutInFeatures = None
        if not(inFeatures is None):
            npOutInFeatures = np.array(auxOutInFeatures)
        npOutInLabels = None
        if not(inLabels is None):
            npOutInLabels = np.array(auxOutInLabels)
        return npOutPts, npOutInFeatures, npOutInLabels


    def get_num_models(self):
        """Method to consult the number of models in the dataset.

        Returns:
            numModels (int): Number of models in the dataset.
        """
        return len(self.fileList_)
        
    
    def set_allowed_samplings(self, allowedSamplings):
        """Method to consult the number of models in the dataset.

        Args:
            allowedSamplings (array of ints): Each element of the array determines an allowed sampling protocol
                that will be used to sample the different models. The implemented sampling protocols are:
                - 0: Uniform sampling
                - 1: Split sampling
                - 2: Gradient sampling
                - 3: Lambert sampling
                - 4: Occlusion sampling
        """
        
        # Check the allowed sampling protocols are valid.
        correctSamplings = True
        for sampl in allowedSamplings:
            correctSamplings = correctSamplings and ((sampl >=0) and (sampl < 5))
        if not correctSamplings:
            raise RuntimeError('Invalid sampling protocol')

        # Check that the dataset contains normals when it has to be used with the sampling protocols
        # lambert and occlusion.
        if ((3 in list(allowedSamplings)) or (4 in list(allowedSamplings))) and (not(self.pointNormals_)):
            raise RuntimeError('The dataset should contain normals in order to use the sampling protocols '\
                'lambert and occlusion')

        self.allowedSamplings_ = allowedSamplings
        

    def reset_caches(self):
        """Method to reset the caches.
        """
        self.cacheCurrNumPts = 0
        self.cacheAddedOrder_ = deque([])
        self.cache_ = {}


    def start_iteration(self):
        """Method to start an iteration over the models.
        """
        self.randomSelection_ = self.randomState_.permutation(len(self.fileList_))
        self.iterator_ = 0


    def has_more_batches(self):
        """Method to consult if there are more models to process in this iteration.

        Returns:
            more (bool): Boolean that indicates that there are more models
                to process.
        """
        return self.iterator_ < len(self.randomSelection_)


    def get_next_batch(self, repeatModelInBatch = False):
        """Method to get the next batch of models.
        
        Args:
            repeatModelInBatch (bool): Boolean that indicates if the batch will be filled with
                the same model.            

        Returns:
            numModelInBatch (int): Number of models in the batch.
            accumPts (nx3 np.array): List of points of the batch.
            accumBatchIds (nx1 np.array): List of model indentifiers within the batch 
                for each point.
            accumFeatures (nxm np.array): List of point features.
            accumLabels (nxl np.array): List of point labels. If the dataset does not contain
                point labels, None is returned instead.
            accumCat (numModelInBatchx1 or nx1 np.array): List of categories of each model 
                in the batch. If the dataset was initialize with pointCategories equal to True,
                the category of each model is provided for each point. If the dataset does not
                contain category information, None is returned instead.
            accumPaths (array of strings): List of paths to the models used in the batch.
        """

        accumPts = np.array([])
        accumBatchIds = np.array([])
        accumFeatures = np.array([])
        accumLabels = None
        if self.pointLabels_:
            accumLabels = np.array([])
        accumCat = None
        if self.useCategories_:
            accumCat = np.array([])
        accumPaths = []
        
        numModelInBatch = 0
        numPtsInBatch = 0

        # Iterate over the elements on the batch.
        for i in range(self.batchSize_):
            # Check if there are enough models left.
            if self.iterator_ < len(self.randomSelection_):
            
                # Check if the model fit in the batch.
                currModelIndex = self.randomSelection_[self.iterator_]
                currModel = self.fileList_[currModelIndex]
                currModelNumPts = self.numPts_[currModelIndex]

                # If the number of points of the model are not stored, we load the file and
                # update the number of points of the model.
                loaded = False
                if currModelNumPts == 0:
                    currPts, currNormals, currFeatures, currLabels = self._load_model_(currModel)
                    currModelNumPts = len(currPts)
                    self.numPts_[currModelIndex] = currModelNumPts
                    loaded = True

                # If the batch has a limited number of points, check if the model fits in the batch.
                if (self.maxPtsxBatch_ == 0) or ((numPtsInBatch+currModelNumPts) <= self.maxPtsxBatch_):

                    # Determine the category of the model if it is necesary.
                    currModelCat = None
                    if self.useCategories_:
                        currModelCat = self.categories_[currModelIndex]

                    # Load the current model from disk or the cache.
                    if not loaded:
                        currPts, currNormals, currFeatures, currLabels = self._load_model_(currModel)

                    # Sample the model.
                    samplingProtocol = self.randomState_.choice(self.allowedSamplings_)
                    if samplingProtocol == 0:
                        currPts, currFeatures, currLabels = self._uniform_sampling_(currPts, len(currPts), 
                            self.uniformSelectFirst_, currFeatures, currLabels, self.numPoints_)
                    elif samplingProtocol == 1:
                        currPts, currFeatures, currLabels = self._non_uniform_sampling_split_(currPts, 
                            len(currPts), currFeatures, currLabels, self.numPoints_)
                    elif samplingProtocol == 2:
                        currPts, currFeatures, currLabels = self._non_uniform_sampling_gradient_(currPts, 
                            len(currPts), currFeatures, currLabels, self.numPoints_)
                    elif samplingProtocol == 3:
                        auxView = (self.randomState_.rand(3)*2.0)-1.0
                        auxView = auxView / np.linalg.norm(auxView)
                        currPts, currFeatures, currLabels = self._non_uniform_sampling_lambert_(auxView, 
                            currPts, currNormals, len(currPts), currFeatures, currLabels, self.numPoints_)
                    else:
                        auxView = (self.randomState_.rand(3)*2.0)-1.0
                        auxView = auxView / np.linalg.norm(auxView)
                        currPts, currFeatures, currLabels = self._non_uniform_sampling_occlusion_(auxView, 
                            currPts, currNormals, len(currPts), currFeatures, currLabels, self.numPoints_)


                    # Augment data.
                    if self.augment_:
                        currPts, rotationMatrix = self._augment_data_rot_(currPts, self.augmentMainAxis_, self.augmentSmallRotations_)
                        if self.pointFeatures_:
                            for currAugmentBlock in self.augmentedFeatures_:
                                currFeatures[:,currAugmentBlock:currAugmentBlock+3], _ = self._augment_data_rot_(
                                    currFeatures[:,currAugmentBlock:currAugmentBlock+3], self.augmentMainAxis_, 
                                    self.augmentSmallRotations_, rotationMatrix)
                        if self.pointLabels_:
                            for currAugmentBlock in self.augmentedLabels_:
                                currLabels[:,currAugmentBlock:currAugmentBlock+3], _ = self._augment_data_rot_(
                                    currLabels[:,currAugmentBlock:currAugmentBlock+3], self.augmentMainAxis_, 
                                    self.augmentSmallRotations_, rotationMatrix)

                    # Append the current model to the batch.         
                    accumPts = np.concatenate((accumPts, currPts), axis=0) if accumPts.size else currPts
                    auxBatchIds = np.array([[i] for it in range(len(currPts))])
                    accumBatchIds = np.concatenate((accumBatchIds, auxBatchIds), axis=0) if accumBatchIds.size else auxBatchIds
                    if self.pointFeatures_:
                        accumFeatures = np.concatenate((accumFeatures, currFeatures), axis=0) if accumFeatures.size else currFeatures
                    else:
                        auxFeatures = np.array([[1.0] for it in range(len(currPts))])
                        accumFeatures = np.concatenate((accumFeatures, auxFeatures), axis=0) if accumFeatures.size else auxFeatures
                    if self.pointLabels_:
                        accumLabels = np.concatenate((accumLabels, currLabels), axis=0) if accumLabels.size else currLabels
                    if self.useCategories_:
                        if self.pointCategories_:
                            auxCategories =  np.array([[currModelCat] for it in range(len(currPts))])
                            accumCat = np.concatenate((accumCat,auxCategories), axis=0) if accumCat.size else auxCategories
                        else:
                            accumCat = np.concatenate((accumCat, np.array([currModelCat])), axis=0) if accumCat.size else np.array([currModelCat])
                    accumPaths.append(currModel)
                                
                    # Update the counters and the iterator.
                    numPtsInBatch  += currModelNumPts
                    numModelInBatch += 1
                    if not repeatModelInBatch:
                        self.iterator_ += 1
            
        if repeatModelInBatch:
            self.iterator_ += 1
            
        return numModelInBatch, accumPts, accumBatchIds, accumFeatures, accumLabels, accumCat, accumPaths