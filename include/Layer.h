//
// Created by yang chen on 2018/3/7.
//

#ifndef MINICNN_LAYER_H
#define MINICNN_LAYER_H

#include <memory>
#include <string>
#include <vector>
#include "Tensor.h"

#define DECLARE_LAYER_TYPE static const std::string layerType;
#define DEFINE_LAYER_TYPE(class_type, type_string) const std::string class_type::layerType = type_string;
#define FRIEND_WITH_NETWORK friend class Network;

namespace MiniCNN
{
    enum class State { TRAIN, TEST };

    class Layer
    {
        FRIEND_WITH_NETWORK

    public:
        inline Shape getInputShape() const { return m_inputShape; }
        inline Shape getOutputShape() const { return m_outputShape; }

    protected:
        inline State getState() const { return m_state; }
        inline void setState(const State state) { m_state = state; }

        inline void setInputShape(const Shape shape) { m_inputShape = shape; }
        inline void setOutputShape(const Shape shape) { m_outputShape = shape; }

        inline float getLearningRate() const { return m_learningRate; }
        inline void setLearningRate(float lr) { m_learningRate = lr; }

        inline std::vector<std::shared_ptr<Tensor>> getGradData() const { return m_gradients; }
        inline std::vector<std::shared_ptr<Tensor>> getParams() const { return m_params; }

        virtual void forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next) = 0;
        virtual void backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                            std::shared_ptr<Tensor>& prevGrad, const std::shared_ptr<Tensor>& nextGrad) = 0;

        virtual void solveInnerParams() { m_outputShape = m_inputShape; }

        virtual std::string getLayerType() const = 0;
        virtual std::string save() const { return getLayerType(); }
        virtual void load(const std::string content) {}

    protected:
        State m_state = State::TRAIN;
        Shape m_inputShape;
        Shape m_outputShape;
        float m_learningRate = 0.1f;
        std::vector<std::shared_ptr<Tensor>> m_gradients;
        std::vector<std::shared_ptr<Tensor>> m_params;

    };
}
#endif //MINICNN_LAYER_H