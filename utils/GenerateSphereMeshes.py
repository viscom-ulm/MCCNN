'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file GenerateShpereMeshes.py

    \brief Script to create a ply scene from a point cloud with an sphere 
        for each point.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import argparse
import os
from os import listdir
from os.path import isfile, join
import numpy as np

from PyUtils import visualize_progress

def icosahedron():
    PHI = (1.0 + np.sqrt(5.0)) / 2.0
    sphereLength = np.sqrt(PHI*PHI + 1.0)
    dist1 = PHI/sphereLength
    dist2 = 1.0/sphereLength

    verts = [
          [-dist2,  dist1, 0], [ dist2,  dist1, 0], [-dist2, -dist1, 0], [ dist2, -dist1, 0],
          [0, -dist2, dist1], [0,  dist2, dist1], [0, -dist2, -dist1], [0,  dist2, -dist1],
          [ dist1, 0, -dist2], [ dist1, 0,  dist2], [-dist1, 0, -dist2], [-dist1, 0,  dist2]
    ]

    faces = [
         [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    return verts, faces

def createEdgeIndex(index1, index2, totalVerts):
    if index1 > index2:
        auxVal = index1
        index1 = index2
        index2 = auxVal
    index1 *= totalVerts
    outIndex = index1 + index2
    return outIndex

def subdivide(verts, faces):
    triangles = len(faces)
    edgeMap = dict([])
    currLength = len(verts)
    for faceIndex in xrange(triangles):
        face = faces[faceIndex]
        v0 = verts[face[0]]
        v1 = verts[face[1]]
        v2 = verts[face[2]]
        
        v3EdgeIndex = createEdgeIndex(face[0], face[1], currLength)
        v3Index = -1
        if v3EdgeIndex in edgeMap:
            v3Index = edgeMap[v3EdgeIndex]
        else:
            newVert = np.array([(v0[0]+v1[0])*0.5, (v0[1]+v1[1])*0.5, (v0[2]+v1[2])*0.5])
            length = np.linalg.norm(newVert)
            verts.append([newVert[0]/length, newVert[1]/length, newVert[2]/length])
            edgeMap[v3EdgeIndex] = len(verts) - 1
            v3Index = len(verts) - 1
        
        v4EdgeIndex = createEdgeIndex(face[1], face[2], currLength)
        v4Index = -1
        if v4EdgeIndex in edgeMap:
            v4Index = edgeMap[v4EdgeIndex]
        else:
            newVert = np.array([(v1[0]+v2[0])*0.5, (v1[1]+v2[1])*0.5, (v1[2]+v2[2])*0.5])
            length = np.linalg.norm(newVert)
            verts.append([newVert[0]/length, newVert[1]/length, newVert[2]/length])
            edgeMap[v4EdgeIndex] = len(verts) - 1
            v4Index = len(verts) - 1

        v5EdgeIndex = createEdgeIndex(face[0], face[2], currLength)
        v5Index = -1
        if v5EdgeIndex in edgeMap:
            v5Index = edgeMap[v5EdgeIndex]
        else:
            newVert = np.array([(v0[0]+v2[0])*0.5, (v0[1]+v2[1])*0.5, (v0[2]+v2[2])*0.5])
            length = np.linalg.norm(newVert)
            verts.append([newVert[0]/length, newVert[1]/length, newVert[2]/length])
            edgeMap[v5EdgeIndex] = len(verts) - 1
            v5Index = len(verts) - 1

        faces.append([v3Index, v4Index, v5Index])
        faces.append([face[0], v3Index, v5Index])
        faces.append([v3Index, face[1], v4Index])
        faces[faceIndex] = [v5Index, v4Index, face[2]]

    return verts, faces   

def load_model(modelsPath):
    points = []
    colors = []
    with open(modelsPath, 'r') as modelFile:
        for line in modelFile:
            line = line.replace("\n", "")
            currPoint = line.split(',')
            points.append([float(currPoint[0]), float(currPoint[1]), float(currPoint[2])])
            colors.append([int(currPoint[3]), int(currPoint[4]), int(currPoint[5])])
    return points, colors  
      
def save_model_ply(modelName, points, colors, sphPts, sphFaces, sphScale): 
    coordMax = np.amax(points, axis=0)
    coordMin = np.amin(points, axis=0)
    aabbSize = (1.0/np.amax(coordMax - coordMin))*sphScale

    newModelName = modelName[:-4]+"_spheres.ply"
    with open(newModelName, 'w') as myFile:
        myFile.write("ply\n")
        myFile.write("format ascii 1.0\n")
        myFile.write("element vertex "+ str(len(sphPts)*len(points))+"\n")
        myFile.write("property float x\n")
        myFile.write("property float y\n")
        myFile.write("property float z\n")
        myFile.write("property uchar red\n")
        myFile.write("property uchar green\n")
        myFile.write("property uchar blue\n")
        myFile.write("element face "+ str(len(sphFaces)*len(points))+"\n")
        myFile.write("property list uchar int vertex_index\n")
        myFile.write("end_header\n")

        for point, color in zip(points, colors):
            for currSphPt in sphPts:
                currPtFlt = [aabbSize*currSphPt[0]+point[0], aabbSize*currSphPt[1]+point[1], aabbSize*currSphPt[2]+point[2]]
                myFile.write(str(currPtFlt[0])+" "+str(currPtFlt[1])+" "+str(currPtFlt[2])+" "+str(color[0])+" "+ str(color[1])+ " "+str(color[2])+"\n")

        offset = 0
        for i in range(len(points)):
            for currSphFace in sphFaces:
                myFile.write("3 "+str(currSphFace[0]+offset)+" "+str(currSphFace[1]+offset)+" "+str(currSphFace[2]+offset)+"\n")
            offset += len(sphPts)
        
    myFile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to generate a 3D model with a sphere for each point.')
    parser.add_argument('--inFolder', default='SphereModels', help='Folder of the input/output models (default: SphereModels)')
    parser.add_argument('--sphSub', default=2, type=int, help='Number of subdivisions applied to the sphere models (default: 2)')
    parser.add_argument('--sphScaling', default=0.005, type=float, help='Scaling applied to the sphere models (default: 0.005)')
    args = parser.parse_args()

    files = [f for f in listdir(args.inFolder+"/") if isfile(join(args.inFolder+"/", f))]

    sphPts, sphFaces = icosahedron()
    for i in range(args.sphSub):
        sphPts, sphFaces = subdivide(sphPts, sphFaces)

    iter  = 0
    for currFile in files:
        points, colors = load_model(args.inFolder+"/"+currFile)
        save_model_ply(args.inFolder+"/"+currFile, points, colors, sphPts, sphFaces, args.sphScaling)
        visualize_progress(iter, len(files))
        iter += 1
