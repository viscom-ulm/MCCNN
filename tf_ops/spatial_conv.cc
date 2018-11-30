/////////////////////////////////////////////////////////////////////////////
/// \file spatial_conv.cc
///
/// \brief C++ operations definition to perform a spatial convolution on a
///        batch of point clouds.
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

REGISTER_OP("SpatialConv")
    .Attr("num_out_features: int")
    .Attr("combin: bool")
    .Attr("batch_size: int")
    .Attr("radius: float")
    .Attr("scale_inv: bool")
    .Attr("avg: bool")
    .Input("points: float32")
    .Input("features: float32")
    .Input("batch_ids: int32")
    .Input("pdfs: float32")
    .Input("sample_pts: float32")
    .Input("start_neighs_indexs: int32")
    .Input("neighs_indexs: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Input("weight_hidden_1: float32")
    .Input("bias_hidden_1: float32")
    .Input("weight_hidden_2: float32")
    .Input("bias_hidden_2: float32")
    .Input("weight_out_layer: float32")
    .Input("bias_out_layer: float32")
    .Output("out_features: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int num_out_features;
        TF_RETURN_IF_ERROR(c->GetAttr("num_out_features", &num_out_features));
        shape_inference::ShapeHandle outputDims = c->MakeShape
            ({c->Dim(c->input(4), 0), num_out_features});
        c->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("SpatialConvGrad")
    .Attr("num_out_features: int")
    .Attr("combin: bool")
    .Attr("batch_size: int")
    .Attr("radius: float")
    .Attr("scale_inv: bool")
    .Attr("avg: bool")
    .Input("points: float32")
    .Input("features: float32")
    .Input("batch_ids: int32")
    .Input("pdfs: float32")
    .Input("sample_pts: float32")
    .Input("start_neighs_indexs: int32")
    .Input("neighs_indexs: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Input("weight_hidden_1: float32")
    .Input("bias_hidden_1: float32")
    .Input("weight_hidden_2: float32")
    .Input("bias_hidden_2: float32")
    .Input("weight_out_layer: float32")
    .Input("bias_out_layer: float32")
    .Input("out_features_grad: float32")
    .Output("features_grad: float32")
    .Output("weight_hidden_1_grad: float32")
    .Output("bias_hidden_1_grad: float32")
    .Output("weight_hidden_2_grad: float32")
    .Output("bias_hidden_2_grad: float32")
    .Output("weight_out_layer_grad: float32")
    .Output("bias_out_laye_grad: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(9));
        c->set_output(2, c->input(10));
        c->set_output(3, c->input(11));
        c->set_output(4, c->input(12));
        c->set_output(5, c->input(13));
        c->set_output(6, c->input(14));
        return Status::OK();
    });


void spatialConvCPU(
    bool pAvg,
    bool pScaleInv,
    int pNumNeighbors,
    int pNumInFeatures,
    int pNumOutFeatures,
    int pNumSamples,
    bool pCombin,
    float pRadius,
    const float* pInPoints,
    const int* pBatchIds,
    const float* pInFeatures,
    const float* pPDFs,
    const float* pSamples,
    const int* pStartIndexs,
    const int* pPackedNeighs,
    const float* pAABBMin,
    const float* pAABBMax,
    const float* pWeights1,
    const float* pBiases1,
    const float* pWeights2,
    const float* pBiases2,
    const float* pWeightsOut,
    const float* pBiasesOut,
    float* pOutFeatues);

void spatialConvGradsCPU(
    bool pAvg,
    bool pScaleInv,
    int pNumNeighbors,
    int pNumInFeatures,
    int pNumOutFeatures,
    int pNumSamples,
    int pNumSamples2,
    bool pCombin,
    float pRadius,
    const float* pInPoints,
    const int* pBatchIds,
    const float* pInFeatures,
    const float* pPDFs,
    const float* pSamples,
    const int* pStartIndexs,
    const int* pPackedNeighs,
    const float* pAABBMin,
    const float* pAABBMax,
    const float* pWeights1,
    const float* pBiases1,
    const float* pWeights2,
    const float* pBiases2,
    const float* pWeightsOut,
    const float* pBiasesOut,
    const float* pInOutFeatueGrads,
    float* pOutFeatureGrads,
    float* pWeights1Grads,
    float* pWeight2Grads,
    float* pWeightOutGrads,
    float* pBiases1Grads,
    float* pBiases2Grads,
    float* pBiasesOutGrads);


class SpatialConvOp : public OpKernel {
    public:
        explicit SpatialConvOp(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("num_out_features", &numOutFeatures_));
            OP_REQUIRES(context, numOutFeatures_ > 0, errors::InvalidArgument("SpatialConvOp expects a positive number of output features")); 

            OP_REQUIRES_OK(context, context->GetAttr("combin", &combin_));

            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument("SpatialConvOp expects a positive radius"));  

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));

            OP_REQUIRES_OK(context, context->GetAttr("avg", &avg_));
        }

        void Compute(OpKernelContext* context) override {

            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("SpatialConvOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inFeaturesTensor=context->input(1);
            OP_REQUIRES(context, inFeaturesTensor.dims() == 2 &&
                inFeaturesTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0), errors::InvalidArgument
                ("SpatialConvOp expects as feature inputs the following dimensions (numPoints)"));
            int numInFeatures = inFeaturesTensor.shape().dim_size(1);
            auto inFeaturesFlat = inFeaturesTensor.flat<float>();
            const float* inFeaturesPtr = &(inFeaturesFlat(0));

            const Tensor& batchIdsTensor = context->input(2); 
            OP_REQUIRES(context, batchIdsTensor.dims() == 2 && 
                batchIdsTensor.shape().dim_size(1) == 1 && 
                batchIdsTensor.shape().dim_size(0) == numPoints, errors::InvalidArgument
                ("SpatialConvOp expects correct btch ids"));
            auto batchIdsFlat = batchIdsTensor.flat<int>();
            const int* batchIdsPtr = &(batchIdsFlat(0));

            const Tensor& inPDFsTensor=context->input(3);
            OP_REQUIRES(context, inPDFsTensor.dims() == 2 &&
                inPDFsTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("SpatialConvOp expects as feature inputs the following dimensions (numPoints)"));
            int numNeighs = inPDFsTensor.shape().dim_size(0);
            auto inPDFsTensorFlat = inPDFsTensor.flat<float>();
            const float* inPDFsTensorPtr = &(inPDFsTensorFlat(0));

            const Tensor& inSamplesTensor = context->input(4);
            OP_REQUIRES(context, inSamplesTensor.dims() == 2, errors::InvalidArgument
                ("SpatialConvOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inSamplesTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects points with three components"));
            int numSamples = inSamplesTensor.shape().dim_size(0);
            auto inSamplesFlat = inSamplesTensor.flat<float>();
            const float* inSamplesPtr = &(inSamplesFlat(0));

            //Process start indexs.
            const Tensor& startIndexTensor = context->input(5); 
            OP_REQUIRES(context, startIndexTensor.dims() == 2 && 
                startIndexTensor.shape().dim_size(1) == 1 && 
                startIndexTensor.shape().dim_size(0) == numSamples, errors::InvalidArgument
                ("SpatialConvOp expects a four dimension tensor for the cell indices"));
            auto startIndexTensorFlat = startIndexTensor.flat<int>();
            const int* startIndexTensorPtr = &(startIndexTensorFlat(0));

            //Process packed neighbors.
            const Tensor& packedNeighTensor = context->input(6); 
            OP_REQUIRES(context, packedNeighTensor.dims() == 2 && 
                packedNeighTensor.shape().dim_size(0) == numNeighs &&
                packedNeighTensor.shape().dim_size(1) == 2, errors::InvalidArgument
                ("SpatialConvOp expects a four dimension tensor for the cell indices"));
            auto packedNeighTensorFlat = packedNeighTensor.flat<int>();
            const int* packedNeighTensorPtr = &(packedNeighTensorFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(7);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(8);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvOp expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Process first hidden layer.
            const Tensor& inWeightsHidd1Tensor = context->input(9);
            const Tensor& inBiasHidd1Tensor = context->input(10);
            OP_REQUIRES(context, 
                inWeightsHidd1Tensor.dims() == 2 && inWeightsHidd1Tensor.shape().dim_size(0) == 3 && 
                inBiasHidd1Tensor.dims() == 1 && inWeightsHidd1Tensor.shape().dim_size(1) == inBiasHidd1Tensor.shape().dim_size(0) &&
                inWeightsHidd1Tensor.shape().dim_size(1)%BLOCK_MLP_SIZE == 0, 
                errors::InvalidArgument("SpatialConvOp expects a correct first hidden layer"));
            auto inWeightsHidd1Flat = inWeightsHidd1Tensor.flat<float>();
            auto inBiasHidd1Flat = inBiasHidd1Tensor.flat<float>();
            const float* inWeightsHidd1Ptr = &(inWeightsHidd1Flat(0));
            const float* inBiasHidd1Ptr = &(inBiasHidd1Flat(0));

            //Process second hidden layer.
            const Tensor& inWeightsHidd2Tensor = context->input(11);
            const Tensor& inBiasHidd2Tensor = context->input(12);
            OP_REQUIRES(context, 
                inWeightsHidd2Tensor.dims() == 2 && inWeightsHidd2Tensor.shape().dim_size(0) == BLOCK_MLP_SIZE && 
                inBiasHidd2Tensor.dims() == 1 && inWeightsHidd2Tensor.shape().dim_size(1) == inBiasHidd2Tensor.shape().dim_size(0) &&
                inWeightsHidd2Tensor.shape().dim_size(1) == inWeightsHidd1Tensor.shape().dim_size(1),  
                errors::InvalidArgument("SpatialConvOp expects a correct second hidden layer"));
            auto inWeightsHidd2Flat = inWeightsHidd2Tensor.flat<float>();
            auto inBiasHidd2Flat = inBiasHidd2Tensor.flat<float>();
            const float* inWeightsHidd2Ptr = &(inWeightsHidd2Flat(0));
            const float* inBiasHidd2Ptr = &(inBiasHidd2Flat(0));

            //Process output layer.
            const Tensor& inWeightsOutLayerTensor = context->input(13);
            const Tensor& inBiasOutLayerTensor = context->input(14);
            OP_REQUIRES(context, 
                inWeightsOutLayerTensor.dims() == 2 && inWeightsOutLayerTensor.shape().dim_size(0) == BLOCK_MLP_SIZE && 
                inBiasOutLayerTensor.dims() == 1 && inWeightsOutLayerTensor.shape().dim_size(1) == inBiasOutLayerTensor.shape().dim_size(0), 
                errors::InvalidArgument("SpatialConvOp expects a correct output layer"));
            OP_REQUIRES(context, inWeightsOutLayerTensor.shape().dim_size(1) % numInFeatures == 0, 
                errors::InvalidArgument("SpatialConvOp expects a number of output neurons multiple of the number of features."));
            if(!combin_){
                OP_REQUIRES(context, 
                    inWeightsOutLayerTensor.shape().dim_size(1) == numInFeatures, 
                    errors::InvalidArgument("SpatialConvOp expects the same number of features in the input and the output"));
            }
            auto inWeightsOutLayerFlat = inWeightsOutLayerTensor.flat<float>();
            auto inBiasOutLayerFlat = inBiasOutLayerTensor.flat<float>();
            const float* inWeightsOutLayerPtr = &(inWeightsOutLayerFlat(0));
            const float* inBiasOutLayerPtr = &(inBiasOutLayerFlat(0));

            //Create the output tensors.
            Tensor* outConvFeatures = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numSamples, numOutFeatures_}, &outConvFeatures));
            auto outConvFeaturesFlat = outConvFeatures->flat<float>();
            float* outConvFeaturesPtr = &(outConvFeaturesFlat(0));

            spatialConvCPU(avg_, scaleInv_, numNeighs, numInFeatures, numOutFeatures_, numSamples, combin_, radius_, inPointsPtr, batchIdsPtr,  inFeaturesPtr, inPDFsTensorPtr, inSamplesPtr, 
                startIndexTensorPtr, packedNeighTensorPtr, inAABBMinPtr, inAABBMaxPtr, inWeightsHidd1Ptr, inBiasHidd1Ptr, inWeightsHidd2Ptr, inBiasHidd2Ptr, 
                inWeightsOutLayerPtr, inBiasOutLayerPtr, outConvFeaturesPtr);
        }

    private:

        int     numOutFeatures_;
        bool    combin_;
        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
        bool    avg_;
};

class SpatialConvGradOp : public OpKernel {
    public:
        explicit SpatialConvGradOp(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("num_out_features", &numOutFeatures_));
            OP_REQUIRES(context, numOutFeatures_ > 0, errors::InvalidArgument("SpatialConvGradOp expects a positive number of output features")); 

            OP_REQUIRES_OK(context, context->GetAttr("combin", &combin_));

            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument("SpatialConvGradOp expects a positive radius"));  

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvGradOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));

            OP_REQUIRES_OK(context, context->GetAttr("avg", &avg_));
        }

        void Compute(OpKernelContext* context) override {

            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("SpatialConvGradOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvGradOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inFeaturesTensor=context->input(1);
            OP_REQUIRES(context, inFeaturesTensor.dims() == 2 &&
                inFeaturesTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0), errors::InvalidArgument
                ("SpatialConvGradOp expects as feature inputs the following dimensions (numPoints)"));
            int numInFeatures = inFeaturesTensor.shape().dim_size(1);
            auto inFeaturesFlat = inFeaturesTensor.flat<float>();
            const float* inFeaturesPtr = &(inFeaturesFlat(0));
            
            const Tensor& batchIdsTensor = context->input(2); 
            OP_REQUIRES(context, batchIdsTensor.dims() == 2 && 
                batchIdsTensor.shape().dim_size(1) == 1 && 
                batchIdsTensor.shape().dim_size(0) == numPoints, errors::InvalidArgument
                ("SpatialConvGradOp expects a four dimension tensor for the cell indices"));
            auto batchIdsFlat = batchIdsTensor.flat<int>();
            const int* batchIdsPtr = &(batchIdsFlat(0));

            const Tensor& inPDFsTensor=context->input(3);
            OP_REQUIRES(context, inPDFsTensor.dims() == 2 &&
                inPDFsTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("SpatialConvGradOp expects as feature inputs the following dimensions (numPoints)"));
            int numNeighs = inPDFsTensor.shape().dim_size(0);
            auto inPDFsTensorFlat = inPDFsTensor.flat<float>();
            const float* inPDFsTensorPtr = &(inPDFsTensorFlat(0));

            const Tensor& inSamplesTensor = context->input(4);
            OP_REQUIRES(context, inSamplesTensor.dims() == 2, errors::InvalidArgument
                ("SpatialConvGradOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inSamplesTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvGradOp expects points with three components"));
            int numSamples = inSamplesTensor.shape().dim_size(0);
            auto inSamplesFlat = inSamplesTensor.flat<float>();
            const float* inSamplesPtr = &(inSamplesFlat(0));

            //Process start indexs.
            const Tensor& startIndexTensor = context->input(5); 
            OP_REQUIRES(context, startIndexTensor.dims() == 2 && 
                startIndexTensor.shape().dim_size(1) == 1 && 
                startIndexTensor.shape().dim_size(0) == numSamples, errors::InvalidArgument
                ("SpatialConvGradOp expects a four dimension tensor for the cell indices"));
            auto startIndexTensorFlat = startIndexTensor.flat<int>();
            const int* startIndexTensorPtr = &(startIndexTensorFlat(0));

            //Process packed neighbors.
            const Tensor& packedNeighTensor = context->input(6); 
            OP_REQUIRES(context, packedNeighTensor.dims() == 2 && 
                packedNeighTensor.shape().dim_size(0) == numNeighs &&
                packedNeighTensor.shape().dim_size(1) == 2, errors::InvalidArgument
                ("SpatialConvGradOp expects a four dimension tensor for the cell indices"));
            auto packedNeighTensorFlat = packedNeighTensor.flat<int>();
            const int* packedNeighTensorPtr = &(packedNeighTensorFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(7);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvGradOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(8);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SpatialConvGradOp expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Process first hidden layer.
            const Tensor& inWeightsHidd1Tensor = context->input(9);
            const Tensor& inBiasHidd1Tensor = context->input(10);
            OP_REQUIRES(context, 
                inWeightsHidd1Tensor.dims() == 2 && inWeightsHidd1Tensor.shape().dim_size(0) == 3 && 
                inBiasHidd1Tensor.dims() == 1 && inWeightsHidd1Tensor.shape().dim_size(1) == inBiasHidd1Tensor.shape().dim_size(0) &&
                inWeightsHidd1Tensor.shape().dim_size(1)%BLOCK_MLP_SIZE == 0, 
                errors::InvalidArgument("SpatialConvGradOp expects a correct first hidden layer"));
            auto inWeightsHidd1Flat = inWeightsHidd1Tensor.flat<float>();
            auto inBiasHidd1Flat = inBiasHidd1Tensor.flat<float>();
            const float* inWeightsHidd1Ptr = &(inWeightsHidd1Flat(0));
            const float* inBiasHidd1Ptr = &(inBiasHidd1Flat(0));

            //Process second hidden layer.
            const Tensor& inWeightsHidd2Tensor = context->input(11);
            const Tensor& inBiasHidd2Tensor = context->input(12);
            OP_REQUIRES(context, 
                inWeightsHidd2Tensor.dims() == 2 && inWeightsHidd2Tensor.shape().dim_size(0) == BLOCK_MLP_SIZE && 
                inBiasHidd2Tensor.dims() == 1 && inWeightsHidd2Tensor.shape().dim_size(1) == inBiasHidd2Tensor.shape().dim_size(0) &&
                inWeightsHidd2Tensor.shape().dim_size(1) == inWeightsHidd1Tensor.shape().dim_size(1),  
                errors::InvalidArgument("SpatialConvGradOp expects a correct second hidden layer"));
            auto inWeightsHidd2Flat = inWeightsHidd2Tensor.flat<float>();
            auto inBiasHidd2Flat = inBiasHidd2Tensor.flat<float>();
            const float* inWeightsHidd2Ptr = &(inWeightsHidd2Flat(0));
            const float* inBiasHidd2Ptr = &(inBiasHidd2Flat(0));

            //Process output layer.
            const Tensor& inWeightsOutLayerTensor = context->input(13);
            const Tensor& inBiasOutLayerTensor = context->input(14);
            OP_REQUIRES(context, 
                inWeightsOutLayerTensor.dims() == 2 && inWeightsOutLayerTensor.shape().dim_size(0) == BLOCK_MLP_SIZE && 
                inBiasOutLayerTensor.dims() == 1 && inWeightsOutLayerTensor.shape().dim_size(1) == inBiasOutLayerTensor.shape().dim_size(0), 
                errors::InvalidArgument("SpatialConvGradOp expects a correct output layer"));
            OP_REQUIRES(context, inWeightsOutLayerTensor.shape().dim_size(1) % numInFeatures == 0, 
                errors::InvalidArgument("SpatialConvGradOp expects a number of output neurons multiple of the number of features."));
            if(!combin_){
                OP_REQUIRES(context, 
                    inWeightsOutLayerTensor.shape().dim_size(1) == numInFeatures, 
                    errors::InvalidArgument("SpatialConvGradOp expects the same number of features in the input and the output"));
            }
            auto inWeightsOutLayerFlat = inWeightsOutLayerTensor.flat<float>();
            auto inBiasOutLayerFlat = inBiasOutLayerTensor.flat<float>();
            const float* inWeightsOutLayerPtr = &(inWeightsOutLayerFlat(0));
            const float* inBiasOutLayerPtr = &(inBiasOutLayerFlat(0));

            const Tensor& inOutFeatureGradsTensor = context->input(15);
            OP_REQUIRES(context, inOutFeatureGradsTensor.dims() == 2 &&
                inOutFeatureGradsTensor.shape().dim_size(0) == numSamples &&
                inOutFeatureGradsTensor.shape().dim_size(1) == numOutFeatures_, errors::InvalidArgument
                ("SpatialConvGradOp expects as feature inputs the following dimensions (numPoints)"));
            auto inOutFeatureGradsFlat = inOutFeatureGradsTensor.flat<float>();
            const float* inOutFeatureGradsPtr = &(inOutFeatureGradsFlat(0));

            //Create the output gradients.
            Tensor* featureGradients = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, inFeaturesTensor.shape(), &featureGradients));
            auto featureGradientsFlat = featureGradients->flat<float>();
            float* featureGradientsPtr = &(featureGradientsFlat(0));
            Tensor* weight1Grads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(1, inWeightsHidd1Tensor.shape(), &weight1Grads));
            auto weight1GradsFlat = weight1Grads->flat<float>();
            float* weight1GradsPtr = &(weight1GradsFlat(0));
            Tensor* bias1Grads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(2, inBiasHidd1Tensor.shape(), &bias1Grads));
            auto bias1GradsFlat = bias1Grads->flat<float>();
            float* bias1GradsPtr = &(bias1GradsFlat(0));
            Tensor* weight2Grads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(3, inWeightsHidd2Tensor.shape(), &weight2Grads));
            auto weight2GradsFlat = weight2Grads->flat<float>();
            float* weight2GradsPtr = &(weight2GradsFlat(0));
            Tensor* bias2Grads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(4, inBiasHidd2Tensor.shape(), &bias2Grads));
            auto bias2GradsFlat = bias2Grads->flat<float>();
            float* bias2GradsPtr = &(bias2GradsFlat(0));
            Tensor* weightOutGrads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(5, inWeightsOutLayerTensor.shape(), &weightOutGrads));
            auto weightOutGradsFlat = weightOutGrads->flat<float>();
            float* weightOutGradsPtr = &(weightOutGradsFlat(0));
            Tensor* biasOutGrads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(6, inBiasOutLayerTensor.shape(), &biasOutGrads));
            auto biasOutGradsFlat = biasOutGrads->flat<float>();
            float* biasOutGradsPtr = &(biasOutGradsFlat(0));


            spatialConvGradsCPU(avg_, scaleInv_, numNeighs, numInFeatures, numOutFeatures_, numSamples, numPoints, combin_, radius_, inPointsPtr, batchIdsPtr,  inFeaturesPtr, inPDFsTensorPtr, inSamplesPtr, 
                startIndexTensorPtr, packedNeighTensorPtr, inAABBMinPtr, inAABBMaxPtr, inWeightsHidd1Ptr, inBiasHidd1Ptr, inWeightsHidd2Ptr, inBiasHidd2Ptr, 
                inWeightsOutLayerPtr, inBiasOutLayerPtr, inOutFeatureGradsPtr, featureGradientsPtr, weight1GradsPtr, weight2GradsPtr, weightOutGradsPtr,
                bias1GradsPtr, bias2GradsPtr, biasOutGradsPtr);
        }

    private:

        int     numOutFeatures_;
        bool    combin_;
        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
        bool    avg_;
};

REGISTER_KERNEL_BUILDER(Name("SpatialConv").Device(DEVICE_GPU), SpatialConvOp);
REGISTER_KERNEL_BUILDER(Name("SpatialConvGrad").Device(DEVICE_GPU), SpatialConvGradOp);