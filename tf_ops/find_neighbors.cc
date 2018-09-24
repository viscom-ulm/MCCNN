/////////////////////////////////////////////////////////////////////////////
/// \file find_neighbors.cc
///
/// \brief C++ operation definition to find the neighboring points within a 
///        certain radius.
///
/// \copyright Copyright (c) 2018 Visual Computing group of Ulm University,  
///            Germany. See the LICENSE file at the top-level directory of 
///            this distribution.
///
/// \author pedro hermosilla (pedro-1.hermosilla-casajus@uni-ulm.de)
/////////////////////////////////////////////////////////////////////////////


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

#include "cuda_kernel_utils.h"

using namespace tensorflow;

REGISTER_OP("FindNeighbors")
    .Attr("radius: float")
    .Attr("batch_size: int")
    .Attr("scale_inv: bool")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("points2: float32")
    .Input("cell_indexs: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("start_indexs: int32")
    .Output("neigh_indexs: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(0), 0), 1});
        shape_inference::ShapeHandle outputDims2 = c->MakeShape({-1, 2});
        c->set_output(0, outputDims);
        c->set_output(1, outputDims2);
        return Status::OK();
    });

unsigned int countNeighborsCPU(
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumCells,
    const float pRadius,
    const float* pInPts,
    const int* pInBatchIds,
    const float* pInPts2,
    const int* pCellIndexs,
    const float* pAABBMin,
    const float* pAABBMax,
    int* pStartIndex);

void packNeighborsCPU(
    const bool pScaleInv,
    const int pNumPoints,
    const int pNumNeighbors,
    const int pNumCells,
    const float pRadius,
    const float* pInPts,
    const int* pInBatchIds,
    const float* pInPts2,
    const int* pCellIndexs,
    const float* pAABBMin,
    const float* pAABBMax,
    int* pAuxBuffOffsets,
    int* pAuxBuffOffsets2,
    int* pStartIndexs,
    int* pPackedIndexs);
    
void computeAuxiliarBuffersSize(
    const int pNumPoints,
    int* PBufferSize1,
    int* PBufferSize2);

class FindNeighborsOp : public OpKernel {
    public:
        explicit FindNeighborsOp(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument("FindNeighborsOp expects a positive radius"));      

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("FindNeighborsOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));
        }

        void Compute(OpKernelContext* context) override {
            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("FindNeighborsOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("FindNeighborsOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inBatchTensor=context->input(1);
            OP_REQUIRES(context, inBatchTensor.dims() == 2 &&
                inBatchTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0) &&
                inBatchTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("FindNeighborsOp expects as batch ids input the following dimensions (numPoints)"));
            auto inBatchFlat = inBatchTensor.flat<int>();
            const int* inBatchPtr = &(inBatchFlat(0));

            //Process input points.
            const Tensor& inPointsTensor2 = context->input(2);
            OP_REQUIRES(context, inPointsTensor2.dims() == 2, errors::InvalidArgument
                ("FindNeighborsOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor2.shape().dim_size(1) == 3, errors::InvalidArgument
                ("FindNeighborsOp expects points with three components"));
            int numPoints2 = inPointsTensor2.shape().dim_size(0);
            auto inPointsFlat2 = inPointsTensor2.flat<float>();
            const float* inPointsPtr2 = &(inPointsFlat2(0));

            //Process input cell ids.
            const Tensor& inCellIdsTensor = context->input(3); 
            OP_REQUIRES(context, inCellIdsTensor.dims() == 5 && 
                inCellIdsTensor.shape().dim_size(0) == batchSize_, errors::InvalidArgument
                ("FindNeighborsOp expects a four dimension tensor for the cell indices"));
            int numCells = inCellIdsTensor.shape().dim_size(1);
            auto inCellIdsFlat = inCellIdsTensor.flat<int>();
            const int* inCellIdsPtr = &(inCellIdsFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(4);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("FindNeighborsOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(5);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("FindNeighborsOp expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Create the output tensors.
            Tensor* startIndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{inPointsTensor.shape().dim_size(0), 1}, &startIndexs));
            auto startIndexsFlat = startIndexs->flat<int>();
            int* startIndexsPtr = &(startIndexsFlat(0));

            //Determine the number of neighbors.
            unsigned int numNeighs = countNeighborsCPU(scaleInv_, numPoints, numCells, radius_, 
                inPointsPtr, inBatchPtr, inPointsPtr2, inCellIdsPtr, inAABBMinPtr, inAABBMaxPtr, startIndexsPtr);

            //Create the second output
            Tensor* neighIndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(1, TensorShape{numNeighs, 2}, &neighIndexs));
            auto neighIndexsFlat = neighIndexs->flat<int>();
            int* neighIndexsPtr = &(neighIndexsFlat(0));
            
            //Create the temporal tensors.
            int tmpBuff1Size, tmpBuff2Size;
            computeAuxiliarBuffersSize(numPoints, &tmpBuff1Size, &tmpBuff2Size);
            Tensor tmpBuff1;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{tmpBuff1Size}, &tmpBuff1));
            auto tmpBuff1Flat = tmpBuff1.flat<int>();
            int* tmpBuff1Ptr = &(tmpBuff1Flat(0));
            Tensor tmpBuff2;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{tmpBuff2Size}, &tmpBuff2));
            auto tmpBuff2Flat = tmpBuff2.flat<int>();
            int* tmpBuff2Ptr = &(tmpBuff2Flat(0));

            //Pack neighbors
            packNeighborsCPU(scaleInv_, numPoints, numNeighs, numCells, radius_, 
                inPointsPtr, inBatchPtr, inPointsPtr2, inCellIdsPtr, inAABBMinPtr, inAABBMaxPtr, 
                tmpBuff1Ptr, tmpBuff2Ptr, startIndexsPtr, neighIndexsPtr);
        }

    private:

        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
};

REGISTER_KERNEL_BUILDER(Name("FindNeighbors").Device(DEVICE_GPU), FindNeighborsOp);