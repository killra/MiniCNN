//
// Created by yang chen on 2018/3/7.
//

#include <algorithm>
#include "../include/Tensor.h"

namespace MiniCNN
{
    Tensor::Tensor(const Shape shape):m_shape(shape), m_data(new float[shape.totalSize()]) {}

    Tensor::~Tensor() {}

    void Tensor::setData(const float item)
    {
        std::fill(m_data.get(), m_data.get() + getShape().totalSize(), item);
    }

    void Tensor::clone(Tensor& target)
    {
        target.m_shape = this->m_shape;
        unsigned int length = sizeof(float) * this->m_shape.totalSize();
        memcpy(target.m_data.get(), this->m_data.get(), length);
    }
}

