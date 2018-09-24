/////////////////////////////////////////////////////////////////////////////
/// \file compute_pdf.cc
///
/// \brief C++ operation definition to approximate the probability 
///        distribution function at each sample in the different receptive 
///        fields.
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

REGISTER_OP("ComputePDF")
    .Attr("window: float")
    .Attr("batch_size: int")
    .Attr("radius: float")
    .Attr("scale_inv: bool")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("start_indexs: int32")
    .Input("neigbors: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("pdfs: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(3), 0), 1});
        c->set_output(0, outputDims);
        return Status::OK();
    });

void computeDPFsCPU(
    const bool pScaleInv,
    const float pWindow, 
    const int numSamples,
    const int pNumNeighbors,
    const float pRadius,
    const float* pInPts,
    const int* pInBatchIds,
    const float* pAABBMin,
    const float* pAABBMax,
    const int* pStartIndexs,
    const int* pPackedIndexs,
    float* pPDFs);

class ComputePDFOp : public OpKernel {
    public:
        explicit ComputePDFOp(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("window", &window_));
            OP_REQUIRES(context, window_ > 0.0, errors::InvalidArgument("ComputePDFOp expects a positive window"));    

            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument("ComputePDFOp expects a positive radius"));  

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("ComputePDFOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));
        }

        void Compute(OpKernelContext* context) override {
            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("ComputePDFOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("ComputePDFOp expects points with three components"));
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

            //Process start indexs.
            const Tensor& startIndexTensor = context->input(2); 
            OP_REQUIRES(context, startIndexTensor.dims() == 2 && 
                startIndexTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("ComputePDFOp expects a four dimension tensor for the cell indices"));
            int numSamples = startIndexTensor.shape().dim_size(0);
            auto startIndexTensorFlat = startIndexTensor.flat<int>();
            const int* startIndexTensorPtr = &(startIndexTensorFlat(0));

            //Process packed neighbors.
            const Tensor& packedNeighTensor = context->input(3); 
            OP_REQUIRES(context, packedNeighTensor.dims() == 2 && 
                packedNeighTensor.shape().dim_size(1) == 2, errors::InvalidArgument
                ("ComputePDFOp expects a four dimension tensor for the cell indices"));
            int numNeighs = packedNeighTensor.shape().dim_size(0);
            auto packedNeighTensorFlat = packedNeighTensor.flat<int>();
            const int* packedNeighTensorPtr = &(packedNeighTensorFlat(0));

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
            Tensor* pdfs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numNeighs, 1}, &pdfs));
            auto pdfsFlat = pdfs->flat<float>();
            float* pdfsPtr = &(pdfsFlat(0));

            //Compute the pdfs
            computeDPFsCPU(scaleInv_, window_, numSamples, numNeighs, radius_, inPointsPtr, inBatchPtr, inAABBMinPtr, inAABBMaxPtr, 
                startIndexTensorPtr, packedNeighTensorPtr, pdfsPtr);
        }

    private:

        float   window_;
        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
};

REGISTER_KERNEL_BUILDER(Name("ComputePDF").Device(DEVICE_GPU), ComputePDFOp);