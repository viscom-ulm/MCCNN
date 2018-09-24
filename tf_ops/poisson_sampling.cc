/////////////////////////////////////////////////////////////////////////////
/// \file poisson_sampling.cc
///
/// \brief C++ operations definition to perform a poisson disk 
///        sampling on a batch of point clouds (O(n)), to obtain the  
///        associated features to  the selected points, and to propagate the 
///        feature gradients.
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

REGISTER_OP("PoissonSampling")
    .Attr("radius: float")
    .Attr("batch_size: int")
    .Attr("scale_inv: bool")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("cell_indexs: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("out_pts: float32")
    .Output("out_batchs: int32")
    .Output("out_indexs: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({-1, 3});
        c->set_output(0, outputDims);
        shape_inference::ShapeHandle outputDims2 = c->MakeShape({-1, 1});
        c->set_output(1, outputDims2);
        c->set_output(2, outputDims2);
        return Status::OK();
    });

REGISTER_OP("GetSampledFeatures")
    .Input("pts_indexs: int32")
    .Input("features: float32")
    .Output("out_features: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(0), 0), c->Dim(c->input(1), 1)});
        c->set_output(0, outputDims);
        return Status::OK();
    });

REGISTER_OP("GetSampledFeaturesGrad")
    .Input("pts_indexs: int32")
    .Input("in_features: float32")
    .Input("sampled_features_gradients: float32")
    .Output("in_gradients: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(1), 0), c->Dim(c->input(1), 1)});
        c->set_output(0, outputDims);
        return Status::OK();
    });

int samplePointCloud(
    const bool scaleInv, 
    const float pRadius,
    const int pNumPoints,
    const int pBatchSize,
    const int pNumCells,
    const float* pAABBMin,
    const float* pAABBMax,
    const float* pPoints,
    const int* pBatchIds,
    const int* pCellIndexs,
    float* pSelectedPts,
    int* pSelectedBatchIds,
    int* pSelectedIndexs,
    bool* pAuxBoolBuffer);

void copyPoints(
    float* pSelectedPts,
    int* pSelectedBatchIds,
    int* pSelectedIndexs,
    const int pNumPts,
    float* pDestPts,
    int* pDestBatchIds,
    int* pDestIndexs);

void getFeaturesSampledPoints(
    int pNumPoints, 
    int pNumFeatures, 
    int pNumSampledPoints, 
    const int* pInPointsIndexs, 
    const float* pInFeature, 
    float* pOutSelFeatures);

void getFeaturesSampledPointsGradients(
    int pNumPoints, 
    int pNumFeatures, 
    int pNumSampledPoints, 
    const int* pInPointsIndexs, 
    const float* pInOutFeatureGrad, 
    float* pOutInFeaturesGradients);

class PoissonSamplingOp : public OpKernel {
    public:
        explicit PoissonSamplingOp(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0.0, errors::InvalidArgument("PoissonSamplingOp expects a positive radius")); 

            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("PoissonSamplingOp expects a positive batch size")); 

            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));
        }

        void Compute(OpKernelContext* context) override {

            //Process input points.
            const Tensor& inPointsTensor = context->input(0);
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("PoissonSamplingOp expects points with the following dimensions (batchSize, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("PoissonSamplingOp expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inBatchTensor=context->input(1);
            OP_REQUIRES(context, inBatchTensor.dims() == 2 &&
                inBatchTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0) &&
                inBatchTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("PoissonSamplingOp expects as batch ids input the following dimensions (numPoints)"));
            auto inBatchFlat = inBatchTensor.flat<int>();
            const int* inBatchPtr = &(inBatchFlat(0));

            //Process input cell ids.
            const Tensor& inCellIdsTensor = context->input(2); 
            OP_REQUIRES(context, inCellIdsTensor.dims() == 5 && 
                inCellIdsTensor.shape().dim_size(0) == batchSize_, errors::InvalidArgument
                ("PoissonSamplingOp expects a four dimension tensor for the cell indices"));
            int numCells = inCellIdsTensor.shape().dim_size(1);
            auto inCellIdsFlat = inCellIdsTensor.flat<int>();
            const int* inCellIdsPtr = &(inCellIdsFlat(0));
            
            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(3);
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("PoissonSamplingOp expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(4);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("PoissonSamplingOp expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Create the temp tensors
            Tensor tmpPts;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{numPoints,3}, &tmpPts));
            auto tmpPtsFlat = tmpPts.flat<float>();
            float* tmpPtsPtr = &(tmpPtsFlat(0));
            Tensor tmpBatchs;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{numPoints,1}, &tmpBatchs));
            auto tmpBatchsFlat = tmpBatchs.flat<int>();
            int* tmpBatchsPtr = &(tmpBatchsFlat(0));
            Tensor tmpIndexs;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{numPoints,1}, &tmpIndexs));
            auto tmpIndexsFlat = tmpIndexs.flat<int>();
            int* tmpIndexsPtr = &(tmpIndexsFlat(0));
            Tensor tmpUsedBool;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<bool>::value,TensorShape{numPoints,1}, &tmpUsedBool));
            auto tmpUsedBoolFlat = tmpUsedBool.flat<bool>();
            bool* tmpUsedBoolPtr = &(tmpUsedBoolFlat(0));

            //Sample the point cloud
            int numSelSamples = samplePointCloud(scaleInv_, radius_, numPoints, batchSize_, numCells, inAABBMinPtr, inAABBMaxPtr,
                inPointsPtr, inBatchPtr, inCellIdsPtr, tmpPtsPtr, tmpBatchsPtr, tmpIndexsPtr, tmpUsedBoolPtr);

            //Create the output tensors.
            Tensor* outSelPts = nullptr;
            Tensor* outSelBatchIds = nullptr;
             Tensor* outSelIndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numSelSamples, 3}, &outSelPts));
            OP_REQUIRES_OK(context,context->allocate_output(1, TensorShape{numSelSamples, 1}, &outSelBatchIds));
            OP_REQUIRES_OK(context,context->allocate_output(2, TensorShape{numSelSamples}, &outSelIndexs));
            auto outSelPtsFlat = outSelPts->flat<float>();
            auto outSelBatchIdsFlat = outSelBatchIds->flat<int>();
            auto outSelIndexsFlat = outSelIndexs->flat<int>();
            float* outSelPtsPtr = &(outSelPtsFlat(0));
            int* outSelBatchIdsPtr = &(outSelBatchIdsFlat(0));
            int* outSelIndexsPtr = &(outSelIndexsFlat(0));

            //Copy the points.
            copyPoints(tmpPtsPtr, tmpBatchsPtr, tmpIndexsPtr, numSelSamples, outSelPtsPtr, outSelBatchIdsPtr, outSelIndexsPtr);
        }

    private:

        float   radius_;
        int     batchSize_;
        bool    scaleInv_;
};

class GetSampledFeaturesOp : public OpKernel {
    public:
        explicit GetSampledFeaturesOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {

            //Process input point indexs.
            const Tensor& inPointsIndexsTensor = context->input(0);
            OP_REQUIRES(context, inPointsIndexsTensor.dims() == 1, errors::InvalidArgument
                ("GetSampledFeaturesOp expects point indexs with the following dimensions (numpoints)"));
            int numSampledPoints = inPointsIndexsTensor.shape().dim_size(0);
            auto inPointsIndexsFlat = inPointsIndexsTensor.flat<int>();
            const int* inPointsIndexsPtr = &(inPointsIndexsFlat(0));

            //Process input features.
            const Tensor& inFeatureTensor=context->input(1);
            OP_REQUIRES(context, inFeatureTensor.dims() == 2, errors::InvalidArgument
                ("GetSampledFeaturesOp features in the right format"));
            int numPoints = inFeatureTensor.shape().dim_size(0);
            int numFeatures = inFeatureTensor.shape().dim_size(1);
            auto inFeatureFlat = inFeatureTensor.flat<float>();
            const float* inFeaturePtr = &(inFeatureFlat(0));

            //Create the output tensors.
            Tensor* outSelFeatures = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numSampledPoints, numFeatures}, &outSelFeatures));
            auto outSelFeaturesFlat = outSelFeatures->flat<float>();
            float* outSelFeaturesPtr = &(outSelFeaturesFlat(0));

            //Get features.
            getFeaturesSampledPoints(numPoints, numFeatures, numSampledPoints, inPointsIndexsPtr, inFeaturePtr, outSelFeaturesPtr);
        }
};

class GetSampledFeaturesGradOp : public OpKernel {
    public:
        explicit GetSampledFeaturesGradOp(OpKernelConstruction* context) : OpKernel(context) {}

        void Compute(OpKernelContext* context) override {

            //Process input point indexs.
            const Tensor& inPointsIndexsTensor = context->input(0);
            OP_REQUIRES(context, inPointsIndexsTensor.dims() == 1, errors::InvalidArgument
                ("GetSampledFeaturesOp expects point indexs with the following dimensions (numpoints)"));
            int numSampledPoints = inPointsIndexsTensor.shape().dim_size(0);
            auto inPointsIndexsFlat = inPointsIndexsTensor.flat<int>();
            const int* inPointsIndexsPtr = &(inPointsIndexsFlat(0));

            //Process input features.
            const Tensor& inFeatureTensor=context->input(1);
            OP_REQUIRES(context, inFeatureTensor.dims() == 2, errors::InvalidArgument
                ("GetSampledFeaturesGradOp expects features in the right format"));
            int numPoints = inFeatureTensor.shape().dim_size(0);
            int numFeatures = inFeatureTensor.shape().dim_size(1);
            auto inFeatureFlat = inFeatureTensor.flat<float>();
            const float* inFeaturePtr = &(inFeatureFlat(0));

            //Process gradients of output features.
            const Tensor& inOutFeatureGradTensor=context->input(2);
            OP_REQUIRES(context, inOutFeatureGradTensor.dims() == 2 &&
                inOutFeatureGradTensor.shape().dim_size(0) == numSampledPoints &&
                inOutFeatureGradTensor.shape().dim_size(1) == numFeatures, errors::InvalidArgument
                ("GetSampledFeaturesGradOp expects gradients in the right format"));
            auto inOutFeatureGradFlat = inOutFeatureGradTensor.flat<float>();
            const float* inOutFeatureGradPtr = &(inOutFeatureGradFlat(0));

            //Create the output tensors.
            Tensor* outInFeaturesGradients = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{numPoints, numFeatures}, &outInFeaturesGradients));
            auto outInFeaturesGradientsFlat = outInFeaturesGradients->flat<float>();
            float* outInFeaturesGradientsPtr = &(outInFeaturesGradientsFlat(0));

            //Get features.
            getFeaturesSampledPointsGradients(numPoints, numFeatures, numSampledPoints, inPointsIndexsPtr, inOutFeatureGradPtr, outInFeaturesGradientsPtr);
        }
};


REGISTER_KERNEL_BUILDER(Name("PoissonSampling").Device(DEVICE_GPU), PoissonSamplingOp);
REGISTER_KERNEL_BUILDER(Name("GetSampledFeatures").Device(DEVICE_GPU), GetSampledFeaturesOp);
REGISTER_KERNEL_BUILDER(Name("GetSampledFeaturesGrad").Device(DEVICE_GPU), GetSampledFeaturesGradOp);