/////////////////////////////////////////////////////////////////////////////
/// \file cuda_kernel_utils.h
///
/// \brief Utilities for the cuda implementations of the tensor operations.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////

#ifndef CUDA_KERNEL_UTILS_H_
#define CUDA_KERNEL_UTILS_H_

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

inline dim3 computeBlockGrid(const unsigned long long int pNumElements, const int pNumThreads)
{
    dim3 finalDimension(pNumElements/pNumThreads, 1, 1);
    finalDimension.x += (pNumElements%pNumThreads!= 0)?1:0;
    while(finalDimension.x >= 65536){
        finalDimension.y *= 2;
        int auxDim = finalDimension.x/2;
        auxDim += (finalDimension.x%2!=0)?1:0;
        finalDimension.x = auxDim;
    }

    while(finalDimension.y >= 65536){
        finalDimension.z *= 2;
        int auxDim = finalDimension.y/2;
        auxDim += (finalDimension.y%2!=0)?1:0;
        finalDimension.y = auxDim;
    }

    return finalDimension;
}

#endif
