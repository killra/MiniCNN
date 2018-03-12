//
// Created by yang chen on 2018/3/9.
//

#ifndef MINICNN_OPTIMIZER_H
#define MINICNN_OPTIMIZER_H

#include <vector>
#include "Tensor.h"

namespace MiniCNN
{
    class Optimizer
    {
    public:
        Optimizer() = default;
        Optimizer(const float lr):m_lr(lr){}
        inline void setLearningRate(const float lr) { m_lr = lr; }
        virtual void update(std::vector<std::shared_ptr<Tensor>> params,
                            const std::vector<std::shared_ptr<Tensor>> gradient) = 0;

    protected:
        float m_lr = 0.0f;
    };

    class SGD : public Optimizer
    {
    public:
        SGD(const float lr) : m_lr(lr){}
        virtual void update(std::vector<std::shared_ptr<Tensor>> params,
                            const std::vector<std::shared_ptr<Tensor>> gradient) override;
    };

    class SGDWithMomentum : public Optimizer
    {
    public:
        SGDWithMomentum(const float lr, const float momentum) : m_lr(lr), m_momentum(momentum){}
        virtual void update(std::vector<std::shared_ptr<Tensor>> params,
                            const std::vector<std::shared_ptr<Tensor>> gradient) override;

    private:
        float m_momentum = 0.0f;
        std::vector<std::shared_ptr<Tensor>> m_historyData;
    };
}

#endif //MINICNN_OPTIMIZER_H