### MCCNN: *Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds*
Created by <a href="https://www.uni-ulm.de/en/in/mi/institute/mi-mitarbeiter/pedro-hermosilla-casajus/" target="_blank">Pedro Hermosilla</a>, <a href="http://www.homepages.ucl.ac.uk/~ucactri/">Tobias Ritschel</a>, <a href="https://www.cs.upc.edu/~ppau/index.html" target="_blank">Pere-Pau Vazquez</a>, <a href="https://www.cs.upc.edu/~alvar/" target="_blank">Alvar Vinacua</a>, <a href="https://www.uni-ulm.de/in/mi/institut/mi-mitarbeiter/tr/" target="_blank">Timo Ropinski</a>.

![teaser](https://github.com/viscom-ulm/MCCNN/blob/master/teaser/Teaser.png)

### Citation
If you find this code useful please consider citing us:

        @article{hermosilla2018mccnn,
          title={Monte Carlo Convolution for Learning on Non-Uniformly Sampled Point Clouds},
          author={Hermosilla, P. and Ritschel, T. and Vazquez, P-P and Vinacua, A. and Ropinski, T.},
          journal={arXiv preprint arXiv:1806.01759},
          year={2018}
        }

### Introduction
We propose an efficient and effective method to learn convolutions for non-uniformly sampled point clouds, as they are obtained with modern acquisition techniques. Learning is enabled by four key novelties: first, representing the convolution kernel itself as a multilayer perceptron; second, phrasing convolution as a Monte Carlo integration problem, third, constructing an unstructured Poisson disk hierarchy for pooling, and fourth, using Monte Carlo convolution as pooling and upsampling operation at different resolutions simultaneously. The key idea across all these contributions is to guarantee adequate consideration of the underlying non-uniform sample distribution function from a Monte Carlo perspective. To make the proposed concepts applicable to real-world tasks, we propose an efficient implementation which significantly reduces the required GPU memory. By employing our method in hierarchical network architectures we can outperform most of the state-of-the-art networks on established point cloud segmentation, classification and normal estimation benchmarks. Furthermore, in contrast to most existing approaches, our method is robust to sampling variations even when only trained on uniformly sampled models.

In this repository, we release the code of our tensor operations and network architectures for classification, segmentation, and normal estimation tasks, which realize the ideas presented in our <a href="https://arxiv.org/abs/1806.01759">arxiv paper</a>. For further details of the techniques implemented here, you can refer to the paper or visit our <a href="https://www.uni-ulm.de/index.php?id=94561">project page</a>.

### Installation
First, install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code presented here was developed using TensorFlow v1.5 GPU version, Python 2.7, and Ubuntu 16.04 TLS. However, it should also work with TensorFlow v1.8 GPU version and Python 3. All the operation were implemented on the GPU, no CPU implementation is provided. Therefore, a workstation with a state-of-the-art GPU is required.

#### Compiling tensorflow operations
In order to train the networks provided in this repository, first, we have to compile the new tensor operations which implement the Monte Carlo convolutions. These operations are located on the folder `tf_ops`. To compile them we should execute the following commands:

    cd tf_ops
    python genCompileScript.py --cudaFolder *path_to_cuda*
    sh compile.sh


### Tasks
The network architectures used for the different tasks can be found on the folder `models`, whilst the scripts to train and evaluate such networks can be found on the folders of the different datasets.  See our <a href="https://arxiv.org/abs/1806.01759">arxiv paper</a> for more details on the network architectures. All scripts can be executed with the argument `--help` to visualize the different options. Some arguments used in our scripts are:

* **grow:** Determines the growth rate of the number of features in the networks. All layers of the networks produce a number of features which is multiple of this number.
* **useDropOut:** If this argument is provided, drop out layers are used in the final MLPs in classification and segmentation networks.
* **dropOutKeepProb:** If this useDropOut is provided, this argument determines the probability to keep the value of a neuron in the MLPs.
* **useDropOutConv:** If this argument is provided, drop out layers are used before each convolution layer in classification and segmentation networks.
* **dropOutKeepProbConv:** If useDropOutConv is provided, this argument determines the probability to keep the value of a feature before each convolution layer.
* **weightDecay:** Scale used in the L2 regularization. If 0.0 is provided, no L2 regularization is performed.
* **ptDropOut:** Probability of selecting a point during loading of the models in the training phase.

#### Classification
We provide 3 different networks for classification tasks (MCClassSmall, MCClass, and MCClassH) which have been tested on the ModelNet40 dataset. We used the resampled ModelNet40 dataset provided in <a href="https://github.com/charlesq34/pointnet2">PointNet++</a>, which contains XYZ position and normals for 10k points per model. Once downloaded, uncompress the zip file inside the folder ModelNet and rename the folder to `data`. Then, you can train and evaluate the different networks. 

**MCClassSmall** This network is composed of only 3 pooling Monte Carlo convolutions. This is the default model used in the classification script and it can be trained and evaluated using the following commands:

    python ModelNet.py --grow 128 --useDropOut
    python ModelNetEval.py --grow 128

**MCClass** This network is composed of a set of Monte Carlo convolutions on the different levels of the point hierarchy. It can be trained and evaluated using the following commands:
    
    python ModelNet.py --model MCClass --useDropOut --useDropOutConv --weightDecay 0.00001
    python ModelNetEval.py --model MCClass

**MCClassH** This network is composed of two different paths which process the different levels of the point hierarchy independently. Whilst one path works directly on the initial point set, the second path works on a lower level on the point hierarchy. It is trained by activating and deactivating these paths in order to be more robust to non-uniformly sampled point clouds. It can be trained and evaluated using the following commands:

    python ModelNet.py --model MCClassH --useDropOut --useDropOutConv 
    python ModelNetEval.py --model MCClassH

#### Segmentation
We provide a network for segmentation tasks (MCSeg) which have been tested on the ShapeNet dataset. We used the resampled ShapeNet dataset provided in <a href="https://github.com/charlesq34/pointnet2">PointNet++</a>, which contains XYZ position, normals and part label per each point. Once downloaded, uncompress the zip file inside the folder ShapeNet and rename the folder to `shape_data`. Then, you can train and evaluate the networks.

**MCSeg** This network has an encoder-decoder architecture, in which the decoder part upsamples features from different levels at the same time. It can be trained and evaluated using the following commands:

    python ShapeNet.py --useDropOut
    python ShapeNetEval.py

The results reported in our paper were obtained by training the network with the following command. However, this configuration requires more GPU memory.

    python ShapeNet.py --grow 128 --useDropOut --useDropOutConv 

#### Normal Estimation
We provide 2 networks for normal estimation tasks (MCNorm, and MCNormSmall) which have been tested on the ModelNet40 dataset. We used the same resampled dataset as the one used in the classification task. Follow the instructions provided in the classification section to download the data. Then, you can train and evaluate the different networks.

**MCNorm:** Network with an encoder-decoder architecture which outputs 3 floats per point. It can be trained and evaluated using the following commands:

    python ModelNetNormals.py
    python ModelNetNormalsEval.py

**MCNormSmall:** Small network composed of only two consecutive Monte Carlo convolutions which output 3 floats per point. This network was designed to evaluate the performance of our convolutions without considering the point hierarchy used. It can be trained and evaluated using the following commands:

    python ModelNetNormals.py --model MCNormSmall
    python ModelNetNormalsEval.py --model MCNormSmall

#### Custom Architectures and Tasks
To design you own network architecture based on our operations, you can take a look first to the `models/MCNormSmall.py` since it is only composed of two convolutions and one can easily see the procedure used to perform these operations. Then, in order to learn how to create a point hierarchy using our Poisson Disk sampling operations and how to transfer features from one level to another, you can open the file `models/MCClassSmall.py`. 

If you want to design your particular tasks, you can base your code on our `ModelNet/ModelNet.py` or `ShapeNet/ShapeNet.py` files. Here you will need to define your own loader for the different models and a loader for the train and test file list.

If you have some doubts about the usage of some operations or scripts you can contact us.

### License
Our code is released under MIT License (see LICENSE file for details).