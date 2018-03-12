//
// Created by yang chen on 2018/3/7.
//

#ifndef MINICNN_FULLYCONNECTEDLAYER_H
#define MINICNN_FULLYCONNECTEDLAYER_H



#include "Layer.h"

namespace MiniCNN
{
    class FullyConnectedLayer : public Layer
    {
        FRIEND_WITH_NETWORK

    public:
        FullyConnectedLayer();
        virtual ~FullyConnectedLayer();

    public:
        void setParameters(const Shape paramShape, const bool enableBias);

    protected:
        DECLARE_LAYER_TYPE;
        virtual void forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next) override;
        virtual void backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                              std::shared_ptr<Tensor>& prevGrad, const std::shared_ptr<Tensor>& nextGrad) override;
        virtual void solveInnerParams() override;
        virtual std::string getLayerType() const override;
        virtual std::string save() const override;
        virtual void load(const std::string content) override;

    private:
        Shape m_paramShape;
        std::shared_ptr<Tensor> m_weight;
        std::shared_ptr<Tensor> m_weightGradient;
        bool m_enableBias = false;
        std::shared_ptr<Tensor> m_bias;
        std::shared_ptr<Tensor> m_biasGradient;
    };
}

#endif //MINICNN_FULLYCONNECTEDLAYER_H