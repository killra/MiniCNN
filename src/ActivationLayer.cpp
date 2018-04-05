//
// Created by yang chen on 2018/3/8.
//

#include "../include/ActivationLayer.h"
#include "../include/CalcFunctions.h"
#include "../include/ThreadPool.h"

namespace MiniCNN
{
    SigmoidLayer::SigmoidLayer() {}
    SigmoidLayer::~SigmoidLayer() {}

    DEFINE_LAYER_TYPE(SigmoidLayer, "SigmoidLayer");
    std::string SigmoidLayer::getLayerType() const
    {
        return layerType;
    }

    void SigmoidLayer::forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next)
    {
        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();
        const float* prevData = prev->getData().get();
        float* nextData = next->getData().get();

        auto worker = [&](const unsigned int start, const unsigned int end)
        {
            const unsigned int offset = start * prevLayerShape.oneBatchSize();
            const unsigned int totalSize = (end - start) * prevLayerShape.oneBatchSize();
            sigmoid(prevData + offset, nextData + offset, totalSize);
        };
        dispatch_worker(worker, prevLayerShape.Batch);
    }

    void SigmoidLayer::backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                                std::shared_ptr<Tensor> &prevGrad, const std::shared_ptr<Tensor> &nextGrad)
    {
        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();
        const float* prevData = prev->getData().get();
        float* nextData = next->getData().get();

        const Shape prevGradShape = prevGrad->getShape();
        const Shape nextGradShape = nextGrad->getShape();
        float* prevGradData = prevGrad->getData().get();
        const float* nextGradData = nextGrad->getData().get();

        prevGrad->setData(0.0f);
        auto worker = [&](const unsigned int start, const unsigned int end)
        {
            const unsigned int offset = start * prevLayerShape.oneBatchSize();
            const unsigned int totalSize = (end - start) * prevLayerShape.oneBatchSize();

            df_sigmoid(nextData + offset, prevGradData + offset, totalSize);
            mul_inplace(prevGradData + offset, nextGradData + offset, totalSize);
        };
        dispatch_worker(worker, prevLayerShape.Batch);
    }

    ReluLayer::ReluLayer() {}
    ReluLayer::~ReluLayer() {}

    DEFINE_LAYER_TYPE(ReluLayer, "ReluLayer");
    std::string ReluLayer::getLayerType() const
    {
        return layerType;
    }

    void ReluLayer::forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next)
    {
        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();
        const float* prevData = prev->getData().get();
        float* nextData = next->getData().get();

        auto worker = [&](const unsigned int start, const unsigned int end)
        {
            const unsigned int offset = start * prevLayerShape.oneBatchSize();
            const unsigned int totalSize = (end - start) * prevLayerShape.oneBatchSize();
            relu(prevData + offset, nextData + offset, totalSize);
        };
        dispatch_worker(worker, prevLayerShape.Batch);
    }

    void ReluLayer::backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                                std::shared_ptr<Tensor> &prevGrad, const std::shared_ptr<Tensor> &nextGrad)
    {
        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();
        const float* prevData = prev->getData().get();
        float* nextData = next->getData().get();

        const Shape prevGradShape = prevGrad->getShape();
        const Shape nextGradShape = nextGrad->getShape();
        float* prevGradData = prevGrad->getData().get();
        const float* nextGradData = nextGrad->getData().get();

        prevGrad->setData(0.0f);
        auto worker = [&](const unsigned int start, const unsigned int end)
        {
            const unsigned int offset = start * prevLayerShape.oneBatchSize();
            const unsigned int totalSize = (end - start) * prevLayerShape.oneBatchSize();

            df_relu(nextData + offset, prevGradData + offset, totalSize);
            mul_inplace(prevGradData + offset, nextGradData + offset, totalSize);
        };
        dispatch_worker(worker, prevLayerShape.Batch);
    }
}