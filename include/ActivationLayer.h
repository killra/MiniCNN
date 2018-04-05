//
// Created by yang chen on 2018/3/8.
//

#ifndef MINICNN_ACTIVATIONLAYER_H
#define MINICNN_ACTIVATIONLAYER_H



#include "Layer.h"

namespace MiniCNN
{
    class ActivationLayer : public Layer
    {};

    class SigmoidLayer : public ActivationLayer
    {
        FRIEND_WITH_NETWORK
    public:
        SigmoidLayer();
        virtual ~SigmoidLayer();
        
    protected:
        DECLARE_LAYER_TYPE;
        virtual std::string getLayerType() const override;
        virtual void forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next) override;
        virtual void backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                              std::shared_ptr<Tensor>& prevGrad, const std::shared_ptr<Tensor>& nextGrad) override;
    };

    class ReluLayer : public ActivationLayer
    {
        FRIEND_WITH_NETWORK
    public:
        ReluLayer();
        virtual ~ReluLayer();
    protected:
        DECLARE_LAYER_TYPE;
        virtual std::string getLayerType() const override;
        virtual void forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next) override;
        virtual void backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                              std::shared_ptr<Tensor>& prevGrad, const std::shared_ptr<Tensor>& nextGrad) override;
    };

}

#endif //MINICNN_ACTIVATIONLAYER_H