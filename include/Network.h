//
// Created by yang chen on 2018/3/9.
//

#ifndef MINICNN_NETWORK_H
#define MINICNN_NETWORK_H

#include <memory>
#include <vector>
#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"

namespace MiniCNN
{
    class Network
    {
    public:
        Network();
        virtual ~Network();

    public:
        void addLayer(std::shared_ptr<Layer> layer);
        void setInputSize(const Shape size);
        void setLossFunction(std::shared_ptr<LossFunction> lossFunction);
        void setOptimizer(std::shared_ptr<Optimizer> optimizer);
        void setLearningRate(const float lr);
        float getLoss(const std::shared_ptr<Tensor> labelTensor, const std::shared_ptr<Tensor> outputTensor);
        float trainBatch(const std::shared_ptr<Tensor> inputTensor, const std::shared_ptr<Tensor> labelTensor);
        std::shared_ptr<Tensor> testBatch(const std::shared_ptr<Tensor> inputTensor);
        bool saveModel(const std::string& modelFile);
        bool loadModel(const std::string& modelFile);

    private:
        State getState() const;
        void setState(State state);

    private:
        std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor> inputTensor);
        float backward(const std::shared_ptr<Tensor> labelTensor);
        std::shared_ptr<Layer> createLayerByType(const std::string layerType);
        std::string getLayerTypeFromLine(const std::string line);

    private:
        State m_state = State::TRAIN;
        std::vector<std::shared_ptr<Layer>> m_layers;
        std::vector<std::shared_ptr<Tensor>> m_data;
        std::vector<std::shared_ptr<Tensor>> m_gradients;
        std::shared_ptr<LossFunction> m_lossFunction;
        std::shared_ptr<Optimizer> m_optimizer;
    };
}

#endif //MINICNN_NETWORK_H