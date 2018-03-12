//
// Created by yang chen on 2018/3/8.
//
#include <sstream>
#include "InputLayer.h"

namespace MiniCNN
{
    InputLayer::InputLayer() {}
    InputLayer::~InputLayer() {}

    DEFINE_LAYER_TYPE(InputLayer, "InputLayer");

    std::string InputLayer::getLayerType() const
    {
        return layerType;
    }

    std::string InputLayer::save() const
    {
        const std::string spliter = " ";
        std::stringstream ss;

        ss << getLayerType() << spliter
           << m_inputShape.Channels << spliter << m_inputShape.Width << spliter << m_inputShape.Height << spliter;
        return ss.str();
    }

    void InputLayer::load(const std::string content)
    {
        std::string _layerType;
        unsigned int channels = 0;
        unsigned int width = 0;
        unsigned int height = 0;
        std::stringstream ss(content);
        ss >> _layerType >> channels >> width >> height;

        Shape shape;
        shape.Batch = 1;
        shape.Channels = channels;
        shape.Width = width;
        shape.Height = height;
        setInputShape(shape);
        setOutputShape(shape);
        solveInnerParams();
    }

    void InputLayer::forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next)
    {
        prev->clone(*next);
    }

    void InputLayer::backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                              std::shared_ptr<Tensor>& prevDiff, const std::shared_ptr<Tensor>& nextDiff)
    {

    }
}