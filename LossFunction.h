//
// Created by yang chen on 2018/3/8.
//

#ifndef MINICNN_LOSSFUNCTIONS_H
#define MINICNN_LOSSFUNCTIONS_H

#include "Tensor.h"

namespace MiniCNN
{
    // TODO:之后记得改名，function不应该命名为类名

    class LossFunction
    {
    public:
        virtual float getLoss(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor) = 0;
        virtual void getGradient(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor,
                            std::shared_ptr<Tensor>& gradient) = 0;
    };

    class CrossEntropyFunction : public LossFunction
    {
    public:
        virtual float getLoss(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor) = 0;
        virtual void getGradient(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor,
                                 std::shared_ptr<Tensor>& gradient) = 0;
    };

    class MSEFunction : public LossFunction
    {
    public:
        virtual float getLoss(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor) = 0;
        virtual void getGradient(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor,
                                 std::shared_ptr<Tensor>& gradient) = 0;
    };
}

#endif //MINICNN_LOSSFUNCTIONS_H