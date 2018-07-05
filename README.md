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

We propose an efficient and effective method to learn convolutions for non-uniformly sampled point clouds, as they are obtained with modern acquisition techniques. Learning is enabled by four key novelties: first, representing the convolution kernel itself as a multilayer perceptron; second, phrasing convolution as a Monte Carlo integration problem, third, constructing an unstructured Poisson disk hierarchy for pooling, and fourth, using Monte Carlo convolution as pooling and upsampling operation at different resolutions simultaneously. The key idea across all these contributions is to guarantee adequate consideration of the underlying non-uniform sample distribution function from a Monte Carlo perspective. To make the proposed concepts applicable for real-world tasks, we propose an efficient implementation which significantly reduces the required GPU memory. By employing our method in hierarchical network architectures we can outperform most of the state-of-the-art networks on established point cloud segmentation, classification and normal estimation benchmarks. Furthermore, in contrast to most existing approaches, our method is robust to sampling variations even when only trained on uniformly sampled models.

In this repository we release the code of our tensor operations and network architectures for classification, segmentation and normal estimation tasks, which realize the ideas presented in our <a href="https://arxiv.org/abs/1806.01759">arxiv paper</a>. For further detials of the techniques implmemented here you can refer to the papaer.

### Installation

First, install <a href="https://www.tensorflow.org/install/">TensorFlow</a>. The code presented here was developed using TensorFlow v1.5 GPU version, python 2.7, and Ubuntu 16.04 TLS. However, it should also work with TensorFlow v1.8 GPU version and python 3. All the operation where implemented on GPU, no CPU implementation is provided. Therefore, a workstation with a state-of-the-art GPU is required.

#### Compiling tensorflow operations

In order to train the networks provided in this repository, first we have to compile the new tensor operations which implement the Monte Carlo convolutions. These operations are located on the folder `tf_ops`. To compile them we should execute the following commands:

    cd tf_ops
    python genCompileScript.py --cudaFolder *path_to_cuda*
    sh compile.sh


### Tasks



#### Classification


MCClassSmall:   python ModelNet.py --grow 128 --useDropOut
                python ModelNetEval.py --grow 128

MCClass:        python ModelNet.py --model MCClass --useDropOut --useDropOutConv
                python ModelNetEval.py --model MCClass

MCClassH:       python ModelNet.py --model MCClassH --useDropOut --useDropOutConv
                python ModelNetEval.py --model MCClassH

#### Segmentation


MCSeg:          python ShapeNet.py --useDropOut
                python ShapeNetEval.py

#### Normal Estimation


MCNorm:         python ModelNetNormals.py
                python ModelNetNormalsEval.py

MCNormSmall:    python ModelNetNormals.py --model MCNormSmall
                python ModelNetNormalsEval.py --model MCNormSmall

#### Custom Architectures

