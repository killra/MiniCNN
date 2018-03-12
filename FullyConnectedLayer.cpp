//
// Created by yang chen on 2018/3/7.
//
#include <sstream>
#include "FullyConnectedLayer.h"
#include "CalcFunctions.h"


namespace MiniCNN
{
    FullyConnectedLayer::FullyConnectedLayer() {}
    FullyConnectedLayer::~FullyConnectedLayer() {}

    void FullyConnectedLayer::setParameters(const Shape paramShape, const bool enableBias)
    {
        m_paramShape = paramShape;
        m_enableBias = enableBias;
        setOutputShape(paramShape);
    }

    DEFINE_LAYER_TYPE(FullyConnectedLayer, "FullyConnectedLayer");
    std::string FullyConnectedLayer::getLayerType() const
    {
        return layerType;
    }

    std::string FullyConnectedLayer::save() const
    {
        const std::string spliter = " ";
        std::stringstream ss;

        ss << getLayerType() << spliter << m_paramShape.Batch << spliter << m_paramShape.Channels << spliter
           << m_paramShape.Width << spliter << m_paramShape.Height << spliter << m_enableBias << spliter;

        const auto weightData = m_weight->getData().get();
        const auto weightShape = m_weight->getShape();
        unsigned int totalSize = weightShape.totalSize();
        for (unsigned int i = 0; i < totalSize; i++)
        {
            ss << weightData[i] << spliter;
        }

        if (m_enableBias)
        {
            const auto biasData = m_bias->getData().get();
            const auto biasShape = m_bias->getShape();
            totalSize = biasShape.totalSize();
            for (unsigned int i = 0; i < totalSize; i++)
            {
                ss << biasData[i] << spliter;
            }
        }

        return ss.str();
    }

    void FullyConnectedLayer::load(const std::string content)
    {
        std::stringstream ss;
        std::string _layerType;
        ss >> _layerType >> m_paramShape.Batch >> m_paramShape.Channels >> m_paramShape.Width
           >> m_paramShape.Height >> m_enableBias;

        setOutputShape(m_paramShape);
        solveInnerParams();
        const auto weightData = m_weight->getData().get();
        const auto weightShape = m_weight->getShape();
        unsigned int totalSize = weightShape.totalSize();
        for (unsigned int i = 0; i < totalSize; i++)
        {
            ss >> weightData[i];
        }

        if (m_enableBias)
        {
            const auto biasData = m_bias->getData().get();
            const auto biasShape = m_bias->getShape();
            totalSize = biasShape.totalSize();
            for (unsigned int i = 0; i < totalSize; i++)
            {
                ss >> biasData[i];
            }
        }
    }

    void FullyConnectedLayer::solveInnerParams()
    {
        const Shape inputShape = getInputShape();
        Shape outputShape = getOutputShape();
        outputShape.Batch = inputShape.Batch;
        setOutputShape(outputShape);

        if (m_weight.get() == nullptr)
        {
            // 由于是全连接层，所以weight数等于输入层的cells * 输出层的cells
            unsigned int weightNum = inputShape.oneBatchSize() * outputShape.oneBatchSize();
            m_weight.reset(new Tensor(Shape(1, weightNum, 1, 1)));

            // 默认使用高斯分布初始化
            normal_distribution_init(m_weight->getData().get(), m_weight->getShape().totalSize(), 0.0f, 0.1f);
        }

        if (m_weightGradient.get() == nullptr)
        {
            m_weightGradient.reset(new Tensor(m_weight->getShape()));
            constant_distribution_init(m_weightGradient->getData().get(), m_weightGradient->getShape().totalSize(), 0.0f);
        }

        if (m_enableBias)
        {
            if (m_bias.get() == nullptr)
            {
                m_bias.reset(new Tensor(Shape(1, outputShape.Channels, 1, 1)));
                // 默认初始化bias为0
                constant_distribution_init(m_bias->getData().get(), m_bias->getShape().totalSize(), 0.0f);
            }

            if (m_biasGradient.get() == nullptr)
            {
                m_biasGradient.reset(new Tensor(m_bias->getShape()));
                constant_distribution_init(m_biasGradient->getData().get(), m_biasGradient->getShape().totalSize(), 0.0f);
            }
        }

        m_params.clear();
        m_params.push_back(m_weight);
        m_params.push_back(m_bias);

        m_gradients.clear();
        m_gradients.push_back(m_weightGradient);
        m_gradients.push_back(m_biasGradient);

    }

    void FullyConnectedLayer::forward(const std::shared_ptr<Tensor> prev, std::shared_ptr<Tensor> next)
    {
        // forward过程中，prevLayer相当于上一层，nextLayer相当于当前层

        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();

        const float* prevLayerData = prev->getData().get();
        float* nextLayerData = next->getData().get();

        const float* pWeightData = m_weight->getData().get();
        const float* pBiasData = m_enableBias ? m_bias->getData().get() : nullptr;

        auto worker = [&](const unsigned int start, const unsigned int end)
        {
            fullyConnect(prevLayerData + start * prevLayerShape.oneBatchSize(), pWeightData, pBiasData,
                        nextLayerData + start * nextLayerShape.oneBatchSize(), end - start,
                        prevLayerShape.oneBatchSize(), nextLayerShape.oneBatchSize());
        };

        // 多线程处理
        dispatch_worker(worker, prevLayerShape.Batch);
    }

    void FullyConnectedLayer::backward(std::shared_ptr<Tensor> prev, const std::shared_ptr<Tensor> next,
                                       std::shared_ptr<Tensor> &prevGrad, const std::shared_ptr<Tensor> &nextGrad)
    {
        // backward过程中，prevLayer相当于当前层，而nextLayer相当于后面传递回来的层

        const Shape prevLayerShape = prev->getShape();
        const Shape nextLayerShape = next->getShape();
        const Shape prevGradShape = prevGrad->getShape();
        const Shape nextGradShape = nextGrad->getShape();
        const Shape weightShape = m_weight->getShape();
        const Shape biasShape = m_enableBias? m_bias->getShape() : Shape();

        const float* prevLayerData = prev->getData().get();
        const float* nextLayerData = next->getData().get();
        float* prevGradData = prevGrad->getData().get();
        const float* nextGradData = nextGrad->getData().get();
        const float* weightData = m_weight->getData().get();
        const float* biasData = m_enableBias? m_bias->getData().get(): nullptr;

        // 根据链式法则，计算当前层的gradient
        prevGrad->setData(0.0f);
        auto worker = [&](const unsigned int start, const unsigned int end)
        {
            for (unsigned int pn = start; pn < end; pn++)
            {
                for (unsigned int pidx = 0; pidx < prevGradShape.oneBatchSize(); pidx++)
                {
                    const unsigned int prevGradIdx = pn * prevGradShape.oneBatchSize() + pidx;
                    for (unsigned int nc = 0; nc < nextGradShape.Channels; nc++)
                    {
                        // 例如3X2的全连接，则weight = （1，6，1，1），channels = （11，21，31，12，22，32）
                        // 求当前层第1个节点的loss = w11*loss1 + w12*loss2
                        const unsigned int weightIdx = nc * prevLayerShape.oneBatchSize() + pidx;
                        const unsigned int nextGradIdx = pn * nextGradShape.oneBatchSize() + nc;

                        prevGradData[prevGradIdx] += weightData[weightIdx] * nextGradData[nextGradIdx];
                    }
                }
            }
        };
        dispatch_worker(worker, prevLayerShape.Batch);

        // 更新当前层的weights
        m_weightGradient->setData(0.0f);
        float* weightGradData = m_weightGradient->getData().get();
        for (unsigned int pn = 0; pn < nextLayerShape.Batch; pn++)
        {
            for (unsigned int nc = 0; nc < nextLayerShape.Channels; nc++)
            {
                const unsigned int nextGradIdx = pn * nextGradShape.oneBatchSize() + nc;
                for (unsigned int pidx = 0; pidx < prevGradShape.oneBatchSize(); pidx++)
                {
                    const unsigned int weightGradIdx = nc * prevGradShape.oneBatchSize() + pidx;
                    const unsigned int prevDataIdx = pn * prevLayerShape.oneBatchSize() + pidx;

                    weightGradData[weightGradIdx] += prevLayerData[prevDataIdx] * nextGradData[nextGradIdx];
                }
            }
        }
        div_inplace(weightGradData, (float)nextLayerShape.Batch, weightShape.totalSize());

        // 更新bias
        if (m_enableBias)
        {
            m_biasGradient->setData(0.0f);
            float* biasGradientData = m_biasGradient->getData().get();
            for (unsigned int nn = 0; nn < nextLayerShape.Batch; nn++)
            {
                for (unsigned int biasGradIdx = 0; biasGradIdx < biasShape.oneBatchSize(); biasGradIdx++)
                {
                    biasGradientData[biasGradIdx] += 1.0f * nextLayerData[nn * biasShape.oneBatchSize() + biasGradIdx];
                }
            }
            //div by batch size
            div_inplace(biasGradientData, (float)nextLayerShape.Batch, biasShape.totalSize());
        }

    }
}