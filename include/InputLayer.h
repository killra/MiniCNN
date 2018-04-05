//
// Created by yang chen on 2018/3/8.
//

#ifndef MINICNN_INPUTLAYER_H
#define MINICNN_INPUTLAYER_H

#include "Layer.h"

namespace MiniCNN
{
    class InputLayer : public Layer {
        FRIEND_WITH_NETWORK

    public:
        InputLayer();

        virtual ~InputLayer();

    protected:
        DECLARE_LAYER_TYPE;

        virtual void forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next) override;

        virtual void backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                              std::shared_ptr<Tensor> &prevGrad, const std::shared_ptr<Tensor> &nextGrad) override;

        virtual std::string getLayerType() const override;

        virtual std::string save() const override;

        virtual void load(const std::string content) override;
    };
}

#endif //MINICNN_INPUTLAYER_H