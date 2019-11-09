'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file PyUtils.py

    \brief File with sevaral utils for python applications.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys

def visualize_progress(val, maxVal, description="", barWidth=20):
    """Method to visualize the progress of a process in the console.

    Args:
        val (int): Current step in the process.
        maxVal (int): Maximum numbef of step of the process.
        description (string): String to be displayed at the current step.
        barWidth (int): Size of the progress bar displayed.
    """

    progress = int((val*barWidth) / maxVal)
    progressBar = ['='] * (progress) + ['>'] + ['.'] * (barWidth - (progress+1))
    progressBar = ''.join(progressBar)
    initBar = "%5d/%5d" % (val + 1, maxVal)
    print(initBar + ' [' + progressBar + '] ' + description)
    sys.stdout.flush()

def save_model(modelName, points, labels = None, colors = None, modLabel = 0): 
    """Method to save a model into a txt file.

    Args:
        modelName (string): Path of the model to be saved.
        points (nx3 np.array): List of points of the model.
        labels (nxm np.array): List of point labels.
        colors (nx3 array): Color associated to each label. If None is provided,
            the method will save the labels instead.
        modLabel (int): Integer value that will be used to apply the mod operation
            to each label.
    """

    with open(modelName+".txt", 'w') as myFile:
        for it, point in enumerate(points):

            myFile.write(str(point[0])+",")
            myFile.write(str(point[1])+",")
            myFile.write(str(point[2]))

            if not(labels is None):
                if not(colors is None):
                    currLabel = int(labels[it][0])
                    if modLabel > 0:
                        currLabel = currLabel%modLabel
                    currColor = colors[currLabel]
                    myFile.write(","+str(currColor[0])+",")
                    myFile.write(str(currColor[1])+",")
                    myFile.write(str(currColor[2]))
                else:
                    currLabels = labels[it]
                    for label in currLabels:
                        myFile.write(","+str(label))
            myFile.write("\n")                
        
    myFile.close()