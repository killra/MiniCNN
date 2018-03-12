//
// Created by yang chen on 2018/3/8.
//
#include <cmath>
#include "SoftmaxLayer.h"

namespace MiniCNN
{
    SoftmaxLayer::SoftmaxLayer() {}
    SoftmaxLayer::~SoftmaxLayer() {}

    DEFINE_LAYER_TYPE(SoftmaxLayer, "SoftmaxLayer");
    std::string SoftmaxLayer::getLayerType() const
    {
        return layerType;
    }

    void SoftmaxLayer::forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next)
    {
        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();

        for (unsigned int batchIdx = 0; batchIdx < nextLayerShape.Batch; batchIdx++)
        {
            const float* prevData = prev->getData().get() + batchIdx * prevLayerShape.oneBatchSize();
            float* nextData = next->getData().get() + batchIdx * nextLayerShape.oneBatchSize();

            // 该方法仿照caffe实现，先减去最大值，再处理。实际效果与理论方法一致
            // find max value
            float maxValue = prevData[0];
            for (unsigned int prevDataIdx = 0; prevDataIdx < prevLayerShape.oneBatchSize(); prevDataIdx++)
            {
                if (prevData[prevDataIdx] > maxValue)
                    maxValue = prevData[prevDataIdx];
            }

            // sum
            float sum = 0;
            for (unsigned int prevDataIdx = 0; prevDataIdx < prevLayerShape.oneBatchSize(); prevDataIdx++)
            {
                nextData[prevDataIdx] = std::exp(prevData[prevDataIdx] - maxValue);
                sum += nextData[prevDataIdx];
            }

            // div
            for (unsigned int prevDataIdx = 0; prevDataIdx < prevLayerShape.oneBatchSize(); prevDataIdx++)
            {
                nextData[prevDataIdx] = nextData[prevDataIdx] / sum;
            }
        }
    }

    void SoftmaxLayer::backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                                std::shared_ptr<Tensor> &prevGrad, const std::shared_ptr<Tensor> &nextGrad)
    {
        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();
        const Shape prevGradShape = prevGrad->getShape();
        const Shape nextGradShape = nextGrad->getShape();

        for (unsigned int batchIdx = 0; batchIdx < prevLayerShape.oneBatchSize(); batchIdx++)
        {
            const float* prevData = prev->getData().get() + batchIdx * prevLayerShape.oneBatchSize();
            const float* nextData = next->getData().get() + batchIdx * nextLayerShape.oneBatchSize();
            float* prevGradData = prevGrad->getData().get() + batchIdx * prevGradShape.oneBatchSize();
            const float* nextGradData = nextGrad->getData().get() + batchIdx * nextGradShape.oneBatchSize();

            for (unsigned int prevGradIdx = 0; prevGradIdx < prevGradShape.oneBatchSize(); prevGradIdx++)
            {
                for (unsigned int nextGradIdx = 0; nextGradIdx < nextGradShape.oneBatchSize(); nextGradIdx++)
                {
                    if (nextGradIdx == prevGradIdx)
                    {
                        // 如果j == i，loss' = aj * (1 - aj) * loss
                        prevGradData[prevGradIdx] += nextData[prevGradIdx] * (1.0f - nextData[prevGradIdx]) * nextGradData[nextGradIdx];
                    }
                    else
                    {
                        // 如果j != i, loss' = -aj * ai * loss
                        prevGradData[prevGradIdx] -= nextData[nextGradIdx] * nextData[prevGradIdx] * nextGradData[nextGradIdx];
                    }
                }
            }

        }
    }

}