'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    \file genCompileScript.py

    \brief Python script to generate the compile script for unix systems.

    \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
                Germany. See the LICENSE file at the top-level directory of 
                this distribution.

    \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import argparse
import tensorflow as tf

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate the compile script for the MCCNN operations.')
    parser.add_argument('--cudaFolder', required=True, help='Path to the CUDA folder')
    parser.add_argument('--MLPSize', default=8, type=int, help='Size of the MLPs (default 8)')
    parser.add_argument('--debugInfo', action='store_true', help='Print debug information during execution (default: False)')
    args = parser.parse_args()

    debugString = " -DPRINT_CONV_INFO" if args.debugInfo else ""
    
    with open("compile.sh", "w") as myCompileScript:
        myCompileScript.write(args.cudaFolder+"/bin/nvcc -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" -std=c++11 aabb_gpu.cu -o aabb_gpu.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" -std=c++11 sort_gpu.cu -o sort_gpu.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" -std=c++11 find_neighbors.cu -o find_neighbors.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" -std=c++11 compute_pdf.cu -o compute_pdf.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" -std=c++11 poisson_sampling.cu -o poisson_sampling.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        myCompileScript.write(args.cudaFolder+"/bin/nvcc -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" -std=c++11 spatial_conv.cu -o spatial_conv.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC\n")
        tensorflowInclude = tf.sysconfig.get_include()
        tensorflowLib = tf.sysconfig.get_lib()
        myCompileScript.write("g++ -std=c++11 -DBLOCK_MLP_SIZE="+str(args.MLPSize)+debugString+" spatial_conv.cc poisson_sampling.cc compute_pdf.cc "\
            "find_neighbors.cc sort_gpu.cc aabb_gpu.cc  spatial_conv.cu.o poisson_sampling.cu.o compute_pdf.cu.o  "\
            "find_neighbors.cu.o sort_gpu.cu.o aabb_gpu.cu.o -o MCConv.so -shared -fPIC -I"+tensorflowInclude+" -I"+tensorflowInclude+"/external/nsync/public "\
            "-I"+args.cudaFolder+"/include -lcudart -L "+args.cudaFolder+"/lib64/ -L"+tensorflowLib+" -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0\n")
            
    with open("MCConvModuleSrc", "r") as mySrcPyScript:
        with open("MCConvModule.py", "w") as myDestPyScript:
            for line in mySrcPyScript:
                myDestPyScript.write(line)
            myDestPyScript.write("\n")
            myDestPyScript.write("\n")
            myDestPyScript.write("def get_block_size():\n")
            myDestPyScript.write("    return "+str(args.MLPSize)+"\n")
            myDestPyScript.write("\n")

