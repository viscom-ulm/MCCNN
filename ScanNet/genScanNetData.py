'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file genScanNetData.py

    \brief Code to process the scannet dataset.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import argparse
import os
import json
import gc
from os import listdir
from os.path import isdir, join
import numpy as np
from ply_reader import read_points_binary_ply

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from PyUtils import visualize_progress

def create_raw_to_scannet_label_map(folder, scannetLabels, version):
    if version == 1:
        indexs = [0,6]
        lines = [line.rstrip() for line in open(folder+"/scannet-labels.combined.tsv")]
    elif version == 2:
        indexs = [1,7]
        lines = [line.rstrip() for line in open(folder+"/scannetv2-labels.combined.tsv")]
    else:
        raise RuntimeError('Unrecognized ScanNet version')
    lines = lines[1:]
    rawToScanNet = {}
    for i in range(len(lines)):
        scannetLabelsSet = set(scannetLabels)
        elements = lines[i].split('\t')
        if elements[indexs[1]] not in scannetLabelsSet:
            rawToScanNet[elements[indexs[0]]] = 'unannotated'
        else:
            rawToScanNet[elements[indexs[0]]] = elements[indexs[1]]
    return rawToScanNet


def get_scanned_rooms(folder):
    scannedRooms = [f for f in listdir(folder+"/scans") if isdir(folder+"/scans/"+f)]
    return scannedRooms

    
def load_room(roomName, folder):
    plydata = read_points_binary_ply(folder+"/scans/"+roomName+"/"+roomName+"_vh_clean_2.ply")
    positions = [[currPt[0], currPt[1], currPt[2]] for currPt in plydata]
    if len(plydata[0]) > 7:
        normals = [[currPt[3], currPt[4], currPt[5]] for currPt in plydata]
        colors = [[currPt[6], currPt[7], currPt[8], currPt[9]] for currPt in plydata]
    else:
        normals = []
        colors = [[currPt[3], currPt[4], currPt[5], currPt[6]] for currPt in plydata]
        
    return positions, normals, colors
    

def load_segmentation(roomName, folder):
    with open(folder+"/scans/"+roomName+"/"+roomName+"_vh_clean_2.0.010000.segs.json") as jsondata:
        d = json.load(jsondata)
        seg = d['segIndices']
    objToSegMap = {}
    for i in range(len(seg)):
        if seg[i] not in objToSegMap:
            objToSegMap[seg[i]] = []
        objToSegMap[seg[i]].append(i)
    return objToSegMap
    

def load_labels(roomName, folder, pts, segMap, rawToScanNetMap, scannetLabels, weights):    
    labels = [0 for i in range(len(pts))]
    with open(folder+"/scans/"+roomName+"/"+roomName+".aggregation.json") as jsondata:
        d = json.load(jsondata)
        for x in d['segGroups']:
            label = 'unannotated'
            if x['label'] in rawToScanNetMap:
                label = rawToScanNetMap[x['label']]
            labelId = scannetLabels.index(label)
            for segment in x['segments']:
                for ptId in segMap[segment]:
                    labels[ptId] = labelId
                    weights[labelId] += 1.0
    return labels
    
    
def compute_aabb(pts):
    maxPt = np.array([-10000.0, -10000.0, -10000.0])
    minPt = np.array([10000.0, 10000.0, 10000.0])
    for currPt in pts:
        maxPt = np.maximum(currPt, maxPt)
        minPt = np.minimum(currPt, minPt)
    return minPt, maxPt
    
    
def process_room(folder, outFolder, room, scannetLabels, rawToScanNetMap, weights, aabbSizesVec, numPointsVec):
    pos, normals, colors = load_room(room, folder)
    segMap = load_segmentation(room, folder)
    labels = load_labels(room, folder, pos, segMap, rawToScanNetMap, scannetLabels, weights)
    minPt, maxPt = compute_aabb(pos)
   
    np.save(outFolder+"/"+room+"_pos.npy", pos)
    if len(normals) > 0:
        np.save(outFolder+"/"+room+"_normals.npy", normals)
    np.save(outFolder+"/"+room+"_colors.npy", colors)
    np.save(outFolder+"/"+room+"_labels.npy", labels)
    np.savetxt(outFolder+"/"+room+"_aabb.txt", [minPt, maxPt])
    
    numPointsVec.append(len(pos))
    aabbSizesVec.append(np.amax(maxPt - minPt))
    
    return len(pos)

            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Script to train MCCNN for segmentation tasks (ShapeNet)')
    parser.add_argument('--inFolder', default='data', help='Folder of the input ScanNet data (default: data)')
    parser.add_argument('--outFolder', default='data_mccnn', help='Folder of the output ScanNet data (default: data_mccnn)')
    parser.add_argument('--version', default=1, type=int, help='ScanNet version (default: 1)')
    args = parser.parse_args()
    
    scannetLabels = ['unannotated', 'wall', 'floor', 'chair', 'table', 'desk', 'bed', 'bookshelf', 'sofa', 
        'sink', 'bathtub', 'toilet', 'curtain', 'counter', 'door', 'window', 'shower curtain', 'refridgerator', 
        'picture', 'cabinet', 'otherfurniture']
    weights = [0.0 for i in range(len(scannetLabels))]
    print(scannetLabels)

    outFolder = args.outFolder+"_v"+str(args.version)
    if not os.path.exists(outFolder): os.mkdir(outFolder)

    np.savetxt(outFolder+"/labels.txt", scannetLabels, fmt='%s')
        
    rawToScanNetMap = create_raw_to_scannet_label_map(args.inFolder, scannetLabels, args.version)
    print rawToScanNetMap

    scannedRooms = get_scanned_rooms(args.inFolder)
    
    np.savetxt(outFolder+"/rooms.txt", scannedRooms, fmt='%s')

    numPointsVec = []
    aabbSizeVec = []
    iter = 0
    for room in scannedRooms:
        visualize_progress(iter, len(scannedRooms), room)
        
        numPoints = 0
        try:
            numPoints = process_room(args.inFolder, outFolder, room, scannetLabels, rawToScanNetMap, weights, aabbSizeVec, numPointsVec)
        except Exception, e:
            print(room+'ERROR!!')
            print(str(e))
            
        gc.collect()
        
        visualize_progress(iter, len(scannedRooms), room + " | "+str(numPoints))
        
        iter += 1
        
    sumWeights = 0.0
    for weight in weights[1:]:
        sumWeights += weight
    
    weights1 = [w for w in weights] 
    weights2 = [w for w in weights] 
    for i in range(len(weights)):
        weights1[i] = weights1[i]/sumWeights
        weights2[i] = weights2[i]/(sumWeights+weights[0])
    
    print(weights1)
    print(weights2)
    
    np.savetxt(outFolder+"/weights.txt", [weights1, weights2])
    np.savetxt(outFolder+"/num_points.txt", numPointsVec)
    np.savetxt(outFolder+"/aabb_sizes.txt", aabbSizeVec)