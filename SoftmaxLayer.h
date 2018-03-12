//
// Created by yang chen on 2018/3/8.
//

#ifndef MINICNN_SOFTMAXLAYER_H
#define MINICNN_SOFTMAXLAYER_H

#include "Layer.h"

namespace MiniCNN
{
    class SoftmaxLayer : public Layer
    {
        FRIEND_WITH_NETWORK

    public:
        SoftmaxLayer();
        virtual ~SoftmaxLayer();

    protected:
        DECLARE_LAYER_TYPE;
        virtual std::string getLayerType() const override;
        virtual void forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next) override;
        virtual void backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                              std::shared_ptr<Tensor>& prevGrad, const std::shared_ptr<Tensor>& nextGrad) override;
    };
}

#endif //MINICNN_SOFTMAXLAYER_H