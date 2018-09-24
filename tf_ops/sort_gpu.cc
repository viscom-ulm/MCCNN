/////////////////////////////////////////////////////////////////////////////
/// \file sort_gpu.cc
///
/// \brief C++ operations definition to distribute a batch of point clouds
///        into a set of uniform grids by using the radix sort algorithm,
///        O(n).
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

REGISTER_OP("SortPointsStep1")
    .Attr("batch_size: int")
    .Attr("cell_size: float")
    .Attr("scale_inv: bool")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("keys: int32")
    .Output("index_new_pos: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle outputDims = c->MakeShape({c->Dim(c->input(0), 0)});
        c->set_output(0, outputDims);
        c->set_output(1, outputDims);
        return Status::OK();
    });

REGISTER_OP("SortPointsStep2")
    .Attr("batch_size: int")
    .Attr("cell_size: float")
    .Attr("scale_inv: bool")
    .Input("points: float32")
    .Input("batch_ids: int32")
    .Input("features: float32")
    .Input("keys: int32")
    .Input("index_new_pos: int32")
    .Input("aabb_min: float32")
    .Input("aabb_max: float32")
    .Output("out_points: float32")
    .Output("out_batch_ids: int32")
    .Output("out_features: float32")
    .Output("cell_indexs: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        int batch_size;
        TF_RETURN_IF_ERROR(c->GetAttr("batch_size", &batch_size));
        shape_inference::ShapeHandle outputDims = c->MakeShape({batch_size, -1, -1, -1, 2});
        c->set_output(0, c->input(0));
        c->set_output(1, c->input(1));
        c->set_output(2, c->input(2));
        c->set_output(3, outputDims);
        return Status::OK();
    });

REGISTER_OP("SortPointsStep2Grad")
    .Input("index_new_pos: int32")
    .Input("out_gradient: float32")
    .Input("out_feature_gradient: float32")
    .Output("in_gradient: float32")
    .Output("in_feature_gradient: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        c->set_output(1, c->input(2));
        return Status::OK();
    });

REGISTER_OP("SortFeaturesBack")
    .Input("features: float32")
    .Input("index_new_pos: int32")
    .Output("out_features: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("SortFeaturesBackGrad")
    .Input("index_new_pos: int32")
    .Input("out_feature_gradient: float32")
    .Output("in_feature_gradient: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });

REGISTER_OP("TransformIndexs")
    .Input("curr_indexs: int32")
    .Input("index_new_pos: int32")
    .Output("out_new_indexs: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

int determineNumCells(
    const bool pScaleInv,
    const int pBatchSize,
    const float pCellSize,
    const float* pAABBMin, 
    const float* pAABBMax);
    
void computeAuxiliarBuffersSize(
    const int pBatchSize, 
    const int pNumCells,
    int* PBufferSize1,
    int* PBufferSize2,
    int* PBufferSize3);

void sortPointsStep1GPUKernel(
    const int pNumPoints, 
    const int pBatchSize,
    const int pNumCells,
    const float* pAABBMin, 
    const float* pAABBMax, 
    const float* pPoints,
    const int* pBatchIds,
    int* pAuxBuffCounters,
    int* pAuxBuffOffsets,
    int* pAuxBuffOffsets2,
    int* pKeys,
    int* pNewIndexs);

void sortPointsStep2GPUKernel(
    const int pNumPoints, 
    const int pBatchSize,
    const int pNumFeatures, 
    const int pNumCells,
    const float* pPoints,
    const int* pBatchIds,
    const float* pFeatures,
    const int* pKeys,
    const int* pNewIndexs,
    int* pAuxBuffer,
    float* pOutPoints,
    int* pOutBatchIds,
    float* pOutFeatures,
    int* pOutCellIndexs);

void sortPointsStep2GradGPUKernel(
    const int pNumPoints, 
    const int pNumFeatures,
    const float* pOutGradients,
    const float* pOutFeatureGradients,
    const int* pNewIndexs,
    float* pInGradients, 
    float* pInFeatureGradients);

void sortFeaturesBack(
    const int pNumPoints,
    const int pNumFeatures,
    const float* pInFeatures,
    const int* pIndexs,
    float* pOutFeatures);

void sortFeaturesBackGrad(
    const int pNumPoints,
    const int pNumFeatures,
    const float* pOutFeatureGrads,
    const int* pIndexs,
    float* pInFeatureGrads);

void computeInverseIndexs(
    const int pNumPoints,
    const int* pIndexs,
    int* pOutIndexs);

void transformIndexs(
    const int pNumIndexs, 
    const int pNumPoints, 
    const int* pInStartIndexs, 
    const int* pInNewIndexs, 
    int* pOutIndexs);

class SortPointsStep1Op : public OpKernel {
    public:
        explicit SortPointsStep1Op(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvolutionGradOp expects a positive batch size"));  

            OP_REQUIRES_OK(context, context->GetAttr("cell_size", &cellSize_));
            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_));   
        }

        void Compute(OpKernelContext* context) override {
            //Process input points.
            const Tensor& inPointsTensor = context->input(0); // Numpoints * 3
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("SortPointsStep1Op expects points with the following dimensions (numPoints, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SortPointsStep1Op expects points with at least three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            const Tensor& inBatchTensor=context->input(1);
            OP_REQUIRES(context, inBatchTensor.dims() == 2 &&
                inBatchTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0) &&
                inBatchTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("SortPointsStep1Op expects as batch ids input the following dimensions (numPoints)"));
            auto inBatchFlat = inBatchTensor.flat<int>();
            const int* inBatchPtr = &(inBatchFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(2);   
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SortPointsStep1Op expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(3);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SortPointsStep1Op expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            //Create the output tensors.
            Tensor* outKeys = nullptr;
            Tensor* outNewndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, TensorShape{inPointsTensor.shape().dim_size(0)}, &outKeys));
            OP_REQUIRES_OK(context,context->allocate_output(1, TensorShape{inPointsTensor.shape().dim_size(0)}, &outNewndexs));
            auto outKeysFlat = outKeys->flat<int>();
            auto outNewndexsFlat = outNewndexs->flat<int>();
            int* outKeysPtr = &(outKeysFlat(0));
            int* outNewndexsPtr = &(outNewndexsFlat(0));

            int numCells = determineNumCells(scaleInv_, batchSize_, cellSize_, inAABBMinPtr, inAABBMaxPtr);
            
            //Create the temporal tensors.
            int numElemsBuff1, numElemsBuff2, numElemsBuff3;
            computeAuxiliarBuffersSize(batchSize_, numCells, &numElemsBuff1, &numElemsBuff2, &numElemsBuff3);
            
            Tensor tmpBuff1;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{numElemsBuff1}, &tmpBuff1));
            auto tmpBuff1Flat = tmpBuff1.flat<int>();
            int* tmpBuff1Ptr = &(tmpBuff1Flat(0));
            Tensor tmpBuff2;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{numElemsBuff2}, &tmpBuff2));
            auto tmpBuff2Flat = tmpBuff2.flat<int>();
            int* tmpBuff2Ptr = &(tmpBuff2Flat(0));
            Tensor tmpBuff3;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{numElemsBuff3}, &tmpBuff3));
            auto tmpBuff3Flat = tmpBuff3.flat<int>();
            int* tmpBuff3Ptr = &(tmpBuff3Flat(0));

            sortPointsStep1GPUKernel(numPoints, batchSize_, numCells, 
                                    inAABBMinPtr, inAABBMaxPtr, 
                                    inPointsPtr, inBatchPtr, 
                                    tmpBuff1Ptr, tmpBuff2Ptr, tmpBuff3Ptr,
                                    outKeysPtr, outNewndexsPtr);
        }
    private:
        int batchSize_;
        float cellSize_;
        bool scaleInv_; 
};

class SortPointsStep2Op : public OpKernel {
    public:
        explicit SortPointsStep2Op(OpKernelConstruction* context) : OpKernel(context) 
        {
            OP_REQUIRES_OK(context, context->GetAttr("batch_size", &batchSize_));
            OP_REQUIRES(context, batchSize_ > 0, errors::InvalidArgument("SpatialConvolutionGradOp expects a positive batch size"));

            OP_REQUIRES_OK(context, context->GetAttr("cell_size", &cellSize_));
            OP_REQUIRES_OK(context, context->GetAttr("scale_inv", &scaleInv_)); 
        }

        void Compute(OpKernelContext* context) override {
            //Process input points.
            const Tensor& inPointsTensor = context->input(0); // Numpoints * 3
            OP_REQUIRES(context, inPointsTensor.dims() == 2, errors::InvalidArgument
                ("SortPointsStep2Op expects points with the following dimensions (numPoints, pointComponents)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SortPointsStep2Op expects points with three components"));
            int numPoints = inPointsTensor.shape().dim_size(0);
            auto inPointsFlat = inPointsTensor.flat<float>();
            const float* inPointsPtr = &(inPointsFlat(0));

            //Process input batch ids.
            const Tensor& inBatchTensor=context->input(1);
            OP_REQUIRES(context, inBatchTensor.dims() == 2 &&
                inBatchTensor.shape().dim_size(0) == inPointsTensor.shape().dim_size(0) &&
                inBatchTensor.shape().dim_size(1) == 1, errors::InvalidArgument
                ("SortPointsStep1Op expects as batch ids input the following dimensions (numPoints)"));
            auto inBatchFlat = inBatchTensor.flat<int>();
            const int* inBatchPtr = &(inBatchFlat(0));

            //Process input features.
            const Tensor& inFeaturesTensor = context->input(2); // Numpoints * numFeatures
            OP_REQUIRES(context, inFeaturesTensor.dims() == 2, errors::InvalidArgument
                ("SortPointsStep2Op expects features with the following dimensions (numPoints, numFeatures)"));
            OP_REQUIRES(context, inFeaturesTensor.shape().dim_size(1) > 0, errors::InvalidArgument
                ("SortPointsStep2Op expects features with at least one component"));
            int numFeatures = inFeaturesTensor.shape().dim_size(1);
            auto inFeaturesFlat = inFeaturesTensor.flat<float>();
            const float* inFeaturesPtr = &(inFeaturesFlat(0));

            //Process input keys.
            const Tensor& inKeysTensor = context->input(3); // Numpoints
            OP_REQUIRES(context, inKeysTensor.dims() == 1, errors::InvalidArgument
                ("SortPointsStep2Op expects keys with the following dimensions (numPoints)"));
            OP_REQUIRES(context, inPointsTensor.shape().dim_size(0) == numPoints, errors::InvalidArgument
                ("SortPointsStep2Op expects the same number of keys and points"));
            auto inKeysFlat = inKeysTensor.flat<int>();
            const int* inKeysPtr = &(inKeysFlat(0));

            //Process input new indexs.
            const Tensor& inNewIndexsTensor = context->input(4); // Numpoints
            OP_REQUIRES(context, inNewIndexsTensor.dims() == 1, errors::InvalidArgument
                ("SortPointsStep2Op expects indexs with the following dimensions (numPoints)"));
            OP_REQUIRES(context, inNewIndexsTensor.shape().dim_size(0) == numPoints, errors::InvalidArgument
                ("SortPointsStep2Op expects the same number of indexs and points"));
            auto inNewIndexsFlat = inNewIndexsTensor.flat<int>();
            const int* inNewIndexsPtr = &(inNewIndexsFlat(0));

            //Process input bounding box.
            const Tensor& inAABBMinTensor = context->input(5);   
            OP_REQUIRES(context, inAABBMinTensor.dims() == 2 
                && inAABBMinTensor.shape().dim_size(0) == batchSize_ && inAABBMinTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SortPointsStep2Op expects a minimum point of the bounding box with 3 components"));
            auto inAABBMinFlat = inAABBMinTensor.flat<float>();
            const float* inAABBMinPtr = &(inAABBMinFlat(0));

            const Tensor& inAABBMaxTensor = context->input(6);
            OP_REQUIRES(context, inAABBMaxTensor.dims() == 2  
                && inAABBMaxTensor.shape().dim_size(0) == batchSize_ && inAABBMaxTensor.shape().dim_size(1) == 3, errors::InvalidArgument
                ("SortPointsStep2Op expects a maximum point of the bounding box with 3 components"));
            auto inAABBMaxFlat = inAABBMaxTensor.flat<float>();
            const float* inAABBMaxPtr = &(inAABBMaxFlat(0));

            int numCells = determineNumCells(scaleInv_, batchSize_, cellSize_, inAABBMinPtr, inAABBMaxPtr);

            //Create the output tensors.
            Tensor* outPoints = nullptr;
            Tensor* outBatchIds = nullptr;
            Tensor* outFeatures = nullptr;
            Tensor* outCellIndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, inPointsTensor.shape(), &outPoints));
            OP_REQUIRES_OK(context,context->allocate_output(1, inBatchTensor.shape(), &outBatchIds));
            OP_REQUIRES_OK(context,context->allocate_output(2, inFeaturesTensor.shape(), &outFeatures));
            OP_REQUIRES_OK(context,context->allocate_output(3, TensorShape{batchSize_, numCells, numCells, numCells, 2}, &outCellIndexs));
            auto outPointsFlat = outPoints->flat<float>();
            auto outBatchIdsFlat = outBatchIds->flat<int>();
            auto outFeaturesFlat = outFeatures->flat<float>();
            auto outCellIndexsFlat = outCellIndexs->flat<int>();
            float* outPointsPtr = &(outPointsFlat(0));
            int* outBatchIdsPtr = &(outBatchIdsFlat(0));
            float* outFeaturesPtr = &(outFeaturesFlat(0));
            int* outCellIndexsPtr = &(outCellIndexsFlat(0));
            
            //Create the temporal tensors.
            Tensor tmpBuff;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value,TensorShape{numPoints}, &tmpBuff));
            auto tmpBuffFlat = tmpBuff.flat<int>();
            int* tmpBuffPtr = &(tmpBuffFlat(0));

            sortPointsStep2GPUKernel(numPoints, batchSize_, numFeatures, numCells, inPointsPtr, 
                inBatchPtr, inFeaturesPtr, inKeysPtr, inNewIndexsPtr, tmpBuffPtr,
                outPointsPtr, outBatchIdsPtr, outFeaturesPtr, outCellIndexsPtr);
        }
    private:
        int batchSize_;
        float cellSize_;
        bool scaleInv_;
};

class SortPointsStep2GradOp: public OpKernel{
    public:
        explicit SortPointsStep2GradOp(OpKernelConstruction * context):OpKernel(context){}
    
        void Compute(OpKernelContext * context)override{
            //Process input new indexs.
            const Tensor& inNewIndexsTensor = context->input(0);
            OP_REQUIRES(context, inNewIndexsTensor.dims() == 1, errors::InvalidArgument
                ("SortPointsStep2GradOp expects indexs with the following dimensions (numPoints)"));
            int numPoints = inNewIndexsTensor.shape().dim_size(0);
            auto inNewIndexsFlat = inNewIndexsTensor.flat<int>();
            const int* inNewIndexsPtr = &(inNewIndexsFlat(0));

            //Process input output gradients.
            const Tensor& inOutputGradTensor = context->input(1); // Numpoints * 3
            OP_REQUIRES(context, inOutputGradTensor.dims() == 2, errors::InvalidArgument
                ("SortPointsStep2Op expects gradients with the following dimensions (numPoints, pointComponents)"));
            OP_REQUIRES(context, inOutputGradTensor.shape().dim_size(1) >= 3, errors::InvalidArgument
                ("SortPointsStep2Op expects gradients with at least three components"));
            auto inOutputGradFlat = inOutputGradTensor.flat<float>();
            const float* inOutputGradPtr = &(inOutputGradFlat(0));

            //Process input output feature gradients.
            const Tensor& inOutputFeatureGradTensor = context->input(2); // Numpoints * numFeatures
            OP_REQUIRES(context, inOutputFeatureGradTensor.dims() == 2, errors::InvalidArgument
                ("SortPointsStep2Op expects gradients of features with the following dimensions (numPoints, numFeatures)"));
            OP_REQUIRES(context, inOutputFeatureGradTensor.shape().dim_size(1) > 0 , errors::InvalidArgument
                ("SortPointsStep2Op expects gradients of features with at least one component"));
            int numFeatures = inOutputFeatureGradTensor.shape().dim_size(1);
            auto inOutputFeatureGradFlat = inOutputFeatureGradTensor.flat<float>();
            const float* inOutputFeatureGradPtr = &(inOutputFeatureGradFlat(0));

            //Create the output input gradients.
            Tensor* outInputGrads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, inOutputGradTensor.shape(), &outInputGrads));
            auto outInputGradsFlat = outInputGrads->flat<float>();
            float* outInputGradsPtr = &(outInputGradsFlat(0));

            //Create the output input feature gradients.
            Tensor* outInputFeatureGrads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(1, inOutputFeatureGradTensor.shape(), &outInputFeatureGrads));
            auto outInputFeatureGradsFlat = outInputFeatureGrads->flat<float>();
            float* outInputFeatureGradsPtr = &(outInputFeatureGradsFlat(0));

            sortPointsStep2GradGPUKernel(numPoints, numFeatures, inOutputGradPtr, inOutputFeatureGradPtr, inNewIndexsPtr, 
                outInputGradsPtr, outInputFeatureGradsPtr);
        }
};

class SortFeaturesBackOp: public OpKernel{
    public:
        explicit SortFeaturesBackOp(OpKernelConstruction * context):OpKernel(context){}
    
        void Compute(OpKernelContext * context)override{
            //Process input new indexs.
            const Tensor& inNewIndexsTensor = context->input(1);
            OP_REQUIRES(context, inNewIndexsTensor.dims() == 1, errors::InvalidArgument
                ("SortFeaturesBackOp expects indexs with the following dimensions (numPoints)"));
            int numPoints = inNewIndexsTensor.shape().dim_size(0);
            auto inNewIndexsFlat = inNewIndexsTensor.flat<int>();
            const int* inNewIndexsPtr = &(inNewIndexsFlat(0));

            //Process input features.
            const Tensor& inFeaturesTensor = context->input(0); // Numpoints * numFeatures
            OP_REQUIRES(context, inFeaturesTensor.dims() == 2, errors::InvalidArgument
                ("SortFeaturesBackOp expects features with the following dimensions (numPoints, numFeatures)"));
            OP_REQUIRES(context, inFeaturesTensor.shape().dim_size(1) > 0 , errors::InvalidArgument
                ("SortFeaturesBackOp expects features with at least one component"));
            int numFeatures = inFeaturesTensor.shape().dim_size(1);
            auto inFeaturesFlat = inFeaturesTensor.flat<float>();
            const float* inFeaturesPtr = &(inFeaturesFlat(0));

            //Create the output features.
            Tensor* outFeatures = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, inFeaturesTensor.shape(), &outFeatures));
            auto outFeaturesFlat = outFeatures->flat<float>();
            float* outFeaturesPtr = &(outFeaturesFlat(0));

            sortFeaturesBack(numPoints, numFeatures, inFeaturesPtr, inNewIndexsPtr, outFeaturesPtr);
        }
};

class SortFeaturesBackGradOp: public OpKernel{
    public:
        explicit SortFeaturesBackGradOp(OpKernelConstruction * context):OpKernel(context){}
    
        void Compute(OpKernelContext * context)override{
            //Process input new indexs.
            const Tensor& inNewIndexsTensor = context->input(0);
            OP_REQUIRES(context, inNewIndexsTensor.dims() == 1, errors::InvalidArgument
                ("SortFeaturesBackGradOp expects indexs with the following dimensions (numPoints)"));
            int numPoints = inNewIndexsTensor.shape().dim_size(0);
            auto inNewIndexsFlat = inNewIndexsTensor.flat<int>();
            const int* inNewIndexsPtr = &(inNewIndexsFlat(0));

            //Process input output feature gradients.
            const Tensor& inOutputFeatureGradTensor = context->input(1); // Numpoints * numFeatures
            OP_REQUIRES(context, inOutputFeatureGradTensor.dims() == 2, errors::InvalidArgument
                ("SortFeaturesBackGradOp expects gradients of features with the following dimensions (numPoints, numFeatures)"));
            OP_REQUIRES(context, inOutputFeatureGradTensor.shape().dim_size(1) > 0 , errors::InvalidArgument
                ("SortFeaturesBackGradOp expects gradients of features with at least one component"));
            int numFeatures = inOutputFeatureGradTensor.shape().dim_size(1);
            auto inOutputFeatureGradFlat = inOutputFeatureGradTensor.flat<float>();
            const float* inOutputFeatureGradPtr = &(inOutputFeatureGradFlat(0));

            //Create the output input feature gradients.
            Tensor* outInputFeatureGrads = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, inOutputFeatureGradTensor.shape(), &outInputFeatureGrads));
            auto outInputFeatureGradsFlat = outInputFeatureGrads->flat<float>();
            float* outInputFeatureGradsPtr = &(outInputFeatureGradsFlat(0));

            sortFeaturesBackGrad(numPoints, numFeatures, inOutputFeatureGradPtr, inNewIndexsPtr, outInputFeatureGradsPtr);
        }
};


class TransformIndexsOp: public OpKernel{
    public:
        explicit TransformIndexsOp(OpKernelConstruction * context):OpKernel(context){}
    
        void Compute(OpKernelContext * context)override{
            //Process indexs to transform. 
            const Tensor& inStartIndexsTensor = context->input(0);
            OP_REQUIRES(context, inStartIndexsTensor.dims() == 1, errors::InvalidArgument
                ("TransformIndexsOp expects indexs with the following dimensions (numPoints)"));
            int numIndexs = inStartIndexsTensor.shape().dim_size(0);
            auto inStartIndexsFlat = inStartIndexsTensor.flat<int>();
            const int* inStartIndexsPtr = &(inStartIndexsFlat(0));

            //Process input new indexs.
            const Tensor& inNewIndexsTensor = context->input(1);
            OP_REQUIRES(context, inNewIndexsTensor.dims() == 1, errors::InvalidArgument
                ("TransformIndexsOp expects indexs with the following dimensions (numPoints)"));
            int numPoints = inNewIndexsTensor.shape().dim_size(0);
            auto inNewIndexsFlat = inNewIndexsTensor.flat<int>();
            const int* inNewIndexsPtr = &(inNewIndexsFlat(0));

            //Create the temp tensors
            Tensor tmpIndexs;
            OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<int>::value, inNewIndexsTensor.shape(), &tmpIndexs));
            auto tmpIndexsFlat = tmpIndexs.flat<int>();
            int* tmpIndexsPtr = &(tmpIndexsFlat(0));

            computeInverseIndexs(numPoints, inNewIndexsPtr, tmpIndexsPtr);

            //Create the output input feature gradients.
            Tensor* outIndexs = nullptr;
            OP_REQUIRES_OK(context,context->allocate_output(0, inStartIndexsTensor.shape(), &outIndexs));
            auto outIndexsFlat = outIndexs->flat<int>();
            int* outIndexsPtr = &(outIndexsFlat(0));

            transformIndexs(numIndexs, numPoints, inStartIndexsPtr, tmpIndexsPtr, outIndexsPtr);
        }
};

REGISTER_KERNEL_BUILDER(Name("SortPointsStep1").Device(DEVICE_GPU), SortPointsStep1Op);
REGISTER_KERNEL_BUILDER(Name("SortPointsStep2").Device(DEVICE_GPU), SortPointsStep2Op);
REGISTER_KERNEL_BUILDER(Name("SortPointsStep2Grad").Device(DEVICE_GPU), SortPointsStep2GradOp);
REGISTER_KERNEL_BUILDER(Name("SortFeaturesBack").Device(DEVICE_GPU), SortFeaturesBackOp);
REGISTER_KERNEL_BUILDER(Name("SortFeaturesBackGrad").Device(DEVICE_GPU), SortFeaturesBackGradOp);
REGISTER_KERNEL_BUILDER(Name("TransformIndexs").Device(DEVICE_GPU), TransformIndexsOp);
