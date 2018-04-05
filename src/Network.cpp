//
// Created by yang chen on 2018/3/9.
//
#include <algorithm>
#include <fstream>
#include <sstream>

#include "../include/Network.h"
#include "../include/Layer.h"
#include "../include/InputLayer.h"
#include "../include/FullyConnectedLayer.h"
#include "../include/ActivationLayer.h"
#include "../include/SoftmaxLayer.h"

namespace MiniCNN
{
    Network::Network() {}
    Network::~Network() {}

    void Network::addLayer(std::shared_ptr<Layer> layer)
    {
        m_layers.push_back(layer);

        const std::shared_ptr<Tensor> prev = m_data[m_data.size() - 1];
        const Shape inputShape = prev->getShape();
        layer->setState(m_state);
        layer->setInputShape(inputShape);
        layer->solveInnerParams();

        const Shape outputShape = layer->getOutputShape();
        m_data.push_back(std::make_shared<Tensor>(outputShape));
        m_gradients.push_back(std::make_shared<Tensor>(outputShape));
    }

    void Network::setInputSize(const Shape size)
    {
        m_data.push_back(std::make_shared<Tensor>(size));
        m_gradients.push_back(std::make_shared<Tensor>(size));
    }

    void Network::setLossFunction(std::shared_ptr<LossFunction> lossFunction)
    {
        m_lossFunction = lossFunction;
    }

    void Network::setOptimizer(std::shared_ptr<Optimizer> optimizer)
    {
        m_optimizer = optimizer;
    }

    void Network::setLearningRate(const float lr)
    {
        m_optimizer->setLearningRate(lr);
    }

    float Network::getLoss(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor)
    {
        if (!m_lossFunction)
            return 0.0f;

        return m_lossFunction->getLoss(labelTensor, outputTensor);
    }

    float Network::trainBatch(const std::shared_ptr<Tensor> inputTensor, const std::shared_ptr<Tensor> labelTensor)
    {
        setState(State::TRAIN);
        forward(inputTensor);
        const float loss = backward(labelTensor);
        return loss;
    }

    std::shared_ptr<Tensor> Network::testBatch(const std::shared_ptr<Tensor> inputTensor)
    {
        setState(State::TEST);
        return forward(inputTensor);
    }

    bool Network::saveModel(const std::string &modelFile)
    {
        std::ofstream ofs(modelFile);
        if (!ofs.is_open())
            return false;

        for (const auto& layer : m_layers)
            ofs << layer->save() + "\n";

        return true;
    }

    bool Network::loadModel(const std::string &modelFile)
    {
        std::ifstream ifs(modelFile);
        if (!ifs.is_open())
            return false;

        std::string line;
        std::getline(ifs, line);

        // 加载inputlayer
        std::string layerType = getLayerTypeFromLine(line);
        std::shared_ptr<Layer> layer = createLayerByType(layerType);
        layer->load(line);
        setInputSize(layer->getInputShape());
        addLayer(layer);

        while (!ifs.eof())
        {
            std::getline(ifs, line);
            if (line.size() <= 2)
                continue;

            layerType = getLayerTypeFromLine(line);
            std::shared_ptr<Layer> layer = createLayerByType(layerType);
            const std::shared_ptr<Tensor> prevData = m_data[m_data.size() - 1];
            const Shape inputShape = prevData->getShape();
            layer->setInputShape(inputShape);
            layer->load(line);
            addLayer(layer);
        }

        setState(State::TEST);
        return true;
    }

    State Network::getState() const
    {
        return m_state;
    }

    void Network::setState(State state)
    {
        m_state = state;
    }

    std::shared_ptr<Layer> Network::createLayerByType(const std::string layerType)
    {
        if (layerType == InputLayer::layerType)
        {
            return std::make_shared<InputLayer>();
        }
        else if (layerType == FullyConnectedLayer::layerType)
        {
            return std::make_shared<FullyConnectedLayer>();
        }
        else if (layerType == SoftmaxLayer::layerType)
        {
            return std::make_shared<SoftmaxLayer>();
        }
        else if (layerType == SigmoidLayer::layerType)
        {
            return std::make_shared<SigmoidLayer>();
        }
        else if (layerType == ReluLayer::layerType)
        {
            return std::make_shared<ReluLayer>();
        }
        else
        {
            return nullptr;
        }
    }

    std::string Network::getLayerTypeFromLine(const std::string line)
    {
        std::stringstream ss(line);
        std::string layerType = "unknown";

        ss >> layerType;
        return layerType;
    }

    std::shared_ptr<Tensor> Network::forward(const std::shared_ptr<Tensor> inputTensor)
    {
        const auto oldBatch = m_data[0]->getShape().Batch;
        const auto newBatch = inputTensor->getShape().Batch;

        // 从inputTensor中拷贝数据，并且更新输入Batch的大小
        if (oldBatch != newBatch)
        {
            for (unsigned int i = 0; i < m_data.size(); i++)
            {
                Shape newShape = m_data[i]->getShape();
                newShape.Batch = newBatch;
                m_data[i].reset(new Tensor(newShape));
            }
        }
        inputTensor->clone(*m_data[0]);

        for (unsigned int i = 0; i < m_layers.size(); i++)
        {
            if (i + 1 < m_layers.size())
            {
                m_data[i + 1]->setData(0.0f);
            }
            m_layers[i]->forward(m_data[i], m_data[i + 1]);
        }

        return m_data[m_data.size() - 1];
    }

    float Network::backward(const std::shared_ptr<Tensor> labelTensor)
    {
        const auto lastOutputData = m_data[m_data.size() - 1];
        const float loss = getLoss(labelTensor, lastOutputData);

        // 处理数据对齐问题
        if (m_gradients.size() != m_layers.size() + 1)
        {
            m_gradients.push_back(std::make_shared<Tensor>(labelTensor->getShape()));
        }

        for (unsigned int i = 0; i < m_data.size(); i++)
        {
            if (!(m_gradients[i]->getShape() == m_data[i]->getShape()))
            {
                m_gradients[i].reset(new Tensor(m_data[i]->getShape()));
            }
        }

        if (!(m_gradients[m_gradients.size() - 1]->getShape() == labelTensor->getShape()))
        {
            m_gradients[m_gradients.size() - 1].reset(new Tensor(labelTensor->getShape()));
        }

        // 计算梯度
        m_lossFunction->getGradient(labelTensor, lastOutputData, m_gradients[m_gradients.size() - 1]);

        for (int i = m_layers.size() - 1; i >= 0; i--)
        {
            m_gradients[i]->setData(0.0f);
            m_layers[i]->backward(m_data[i], m_data[i + 1], m_gradients[i], m_gradients[i + 1]);
        }

        // 更新参数
        for (int i = m_layers.size() - 1; i >= 0; i--)
        {
            m_optimizer->update(m_layers[i]->getParams(), m_layers[i]->getGradData());
        }

        return loss;
    }
}