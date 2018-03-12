//
// Created by yang chen on 2018/3/9.
//

#include "Optimizer.h"

namespace MiniCNN
{
    void SGD::update(std::vector<std::shared_ptr<Tensor>> params,
                     const std::vector<std::shared_ptr<Tensor>> gradient)
    {
        // w = w - lr * g
        for (unsigned int i = 0; i < params.size(); i++)
        {
            float* paramData = params[i]->getData().get();
            const float* gradientData = gradient[i]->getData().get();
            for (unsigned int j = 0; j < params[i]->getShape().totalSize(); j++)
            {
                paramData[j] -= m_lr * gradientData[j];
            }
        }
    }

    void SGDWithMomentum::update(std::vector<std::shared_ptr<Tensor>> params,
                                 const std::vector<std::shared_ptr<Tensor>> gradient)
    {
        if (m_historyData.size() != params.size())
        {
            m_historyData.resize(params.size());
        }

        for (unsigned int i = 0; i < params.size(); i++)
        {
            if ((m_historyData[i].get() == nullptr) || !(m_historyData[i]->getShape() == params[i]->getShape()))
            {
                m_historyData[i].reset(new Tensor(params[i]->getShape()));
                m_historyData[i]->setData(0.0f);
            }

            float* paramData = params[i]->getData().get();
            const float* gradientData = gradient[i]->getData().get();
            float* historyData = m_historyData[i]->getData().get();

            for (unsigned int j = 0; j < params[i]->getShape().totalSize(); j++)
            {
                historyData[j] = m_momentum * historyData[j] - m_lr * gradientData[j];
                paramData[j] += historyData[j];
            }

        }
    }
}
