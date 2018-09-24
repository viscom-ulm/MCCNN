/////////////////////////////////////////////////////////////////////////////
/// \file aabb_gpu.cc
///
/// \brief C++ operation definition to compute the axis aligned bounding box 
///        of a batch of point clouds.
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

using namespace tensorflow;

REGISTER_OP("ComputeAabb")
    .Attr("batch_size: int")
    .Attr("scale_inv: bool")
    .Input("points : float32")
    .Input("batch_ids: int32")
    .Output("aabb_min : float32")
    .Output("aabb_max : float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int batch_size;
        TF_RETURN_IF_ERROR(c->GetAttr("batch_size", &batch_size));
        shape_inference::ShapeHandle aabbDims = c->MakeShape({batch_size, 3});
        c->set_output(0, aabbDims);
        c->set_output(1, aabbDims);
        return Status::OK();
    });

void computeAABB(
    const bool pScaleInv, const int pNumPoints, const int pBatchSize, 
    const float* pPoints, const int* pBatchIds, float* pAABBMin, float* pAABBMax);

class ComputeAABBOp : public OpKernel {
    public:
        explicit ComputeAABBOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvolutionGradOp expects a positive batch size"));     

             OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& inPointsTensor=context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("ComputeAabb expects as input the following dimensions (numPoints, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) >= 3, errors::InvalidArgument
                ("ComputeAabb expects points with at least three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            int pointSize = inPointsTensor.shape().dim_size(1);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inBatchTensor=context->input(1);
            OP_REQUIRES(context, inBatchTensor.dims() == 2 &&
                inBatchTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0) &&
                inBatchTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("ComputeAabb expects as batch ids input the following dimensions (numPoints)"));
            auto inBatchFlat = inBatchTensor.flat<int>();
            const int* inBatchPtr = &(inBatchFlat(0));
            
            Tensor* outAABBMin = NULL;
            Tensor* outAABBMax = NULL;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{batchSize_, 3}, &outAABBMin));
            OP_REQUIRES_OK(context,context->allocate_output(1, TensorShape{batchSize_, 3}, &outAABBMax));
            auto outAABBMinFlat = outAABBMin->flat<float>();
            auto outAABBMaxFlat = outAABBMax->flat<float>();
            float* outAABBMinPtr = &(outAABBMinFlat(0));
            float* outAABBMaxPtr = &(outAABBMaxFlat(0));

            computeAABB(scaleInv_, numPoints, batchSize_, inPointsPtr, inBatchPtr, outAABBMinPtr, outAABBMaxPtr);
        }
    private:
        int   batchSize_;
        bool scaleInv_;
};

REGISTER_KERNEL_BUILDER(Name("ComputeAabb").Device(DEVICE_GPU), ComputeAABBOp);