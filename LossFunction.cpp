//
// Created by yang chen on 2018/3/8.
//
#include <cmath>
#include "LossFunction.h"
#include "CalcFunctions.h"

namespace MiniCNN
{
    float CrossEntropyFunction::getLoss(const std::shared_ptr<Tensor> labelTensor,
                                        const std::shared_ptr<Tensor> outputTensor)
    {
        const Shape outputShape = outputTensor->getShape();
        const float* labelData = labelTensor->getData().get();
        const float* outputData = outputTensor->getData().get();
        float loss = 0.0f;

        for (unsigned int i = 0; i < outputShape.totalSize(); i++)
        {
            const float oneLoss = -labelData[i] * std::log(outputData[i]);
            // 这里使用的是统计学中的简单移动平均值法，目的是为了防止数值溢出，其值是近似准确值
            // 该方法在BatchNormalization 的论文中有用到
            loss = moving_average(loss, i+1, oneLoss);
        }
        return loss * outputShape.oneBatchSize();
    }

    void CrossEntropyFunction::getGradient(const std::shared_ptr<Tensor> labelTensor,
                                            const std::shared_ptr<Tensor> outputTensor,
                                            std::shared_ptr<Tensor> &gradient)
    {
        const Shape labelShape = labelTensor->getShape();
        const Shape outputShape = outputTensor->getShape();
        const Shape gradientShape = gradient->getShape();

        gradient->setData(0.0f);
        for (unsigned int batchIdx = 0; batchIdx < outputShape.Batch; batchIdx++)
        {
            const float* labelData = labelTensor->getData().get() + batchIdx * labelShape.oneBatchSize();
            const float* outputData = outputTensor->getData().get() + batchIdx * outputShape.oneBatchSize();
            float* gradientData = gradient->getData().get() + batchIdx * gradientShape.oneBatchSize();

            for (unsigned int gradIdx = 0; gradIdx < gradientShape.oneBatchSize(); gradIdx++)
            {
                gradientData[gradIdx] -= labelData[gradIdx] / outputData[gradIdx];
            }
        }

    }


    float MSEFunction::getLoss(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor)
    {
        const Shape outputShape = outputTensor->getShape();
        const float* labelData = labelTensor->getData().get();
        const float* outputData = outputTensor->getData().get();
        float loss = 0.0f;

        for (unsigned int i = 0; i < outputShape.totalSize(); i++)
        {
            const float oneLoss = (labelData[i] - outputData[i]) * (labelData[i] - outputData[i]);
            loss = moving_average(loss, i+1, oneLoss);
        }
        return loss * outputShape.oneBatchSize();
    }

    void MSEFunction::getGradient(const std::shared_ptr<Tensor> labelTensor,
                                  const std::shared_ptr<Tensor> outputTensor,
                                  std::shared_ptr<Tensor> &gradient)
    {
        const Shape labelShape = labelTensor->getShape();
        const Shape outputShape = outputTensor->getShape();
        const Shape gradientShape = gradient->getShape();

        gradient->setData(0.0f);
        for (unsigned int batchIdx = 0; batchIdx < outputShape.Batch; batchIdx++)
        {
            const float* labelData = labelTensor->getData().get() + batchIdx * labelShape.oneBatchSize();
            const float* outputData = outputTensor->getData().get() + batchIdx * outputShape.oneBatchSize();
            float* gradientData = gradient->getData().get() + batchIdx * gradientShape.oneBatchSize();

            for (unsigned int gradIdx = 0; gradIdx < gradientShape.oneBatchSize(); gradIdx++)
            {
                gradientData[gradIdx] += 2.0f * (outputData[gradIdx] - labelData[gradIdx]);
            }
        }
    }
}