//
// Created by yang chen on 2018/3/7.
//

#ifndef MINICNN_TENSOR_H
#define MINICNN_TENSOR_H

#include <memory>

namespace MiniCNN
{
    class Shape
    {
    public:
        Shape() = default;
        Shape(unsigned int batch, unsigned int channels, unsigned int width, unsigned int height)
                :Batch(batch), Channels(channels), Width(width), Height(height){}

        inline unsigned int totalSize() const { return Batch * Channels * Width * Height;}
        inline unsigned int oneBatchSize() const { return Channels * Width * Height; }
        inline unsigned int oneChannelSize() const { return Width * Height; }

        inline bool operator==(const Shape& other) const
        {
            return (other.Batch == Batch) && (other.Channels == Channels)
                   && (other.Width == Width) && (other.Height == Height);
        }

        inline bool operator!=(const Shape& other) const
        {
            return !(*this == other);
        }

        inline unsigned int getIndex(const unsigned int inBatch, const unsigned int inChannels,
                                     const unsigned int inRow, const unsigned int inCol)
        {
            return inBatch * Channels * Width * Height + inChannels * Width * Height + inRow * Width + inCol;
        }

        inline unsigned int getIndex(const unsigned int inChannels, const unsigned int inRow, const unsigned int inCol)
        {
            return inChannels * Width * Height + inRow * Width + inCol;
        }

        unsigned int Batch = 0;
        unsigned int Channels = 0;
        unsigned int Width = 0;
        unsigned int Height = 0;
    };

    class Tensor
    {
    public:
        Tensor(const Shape shape);
        virtual ~Tensor();

        inline Shape getShape() const { return m_shape; }
        inline std::shared_ptr<float> getData() const {return m_data; }
        void setData(const float item);
        void clone(Tensor& target);

    private:
        Shape m_shape;
        std::shared_ptr<float> m_data;
    };
}

#endif //MINICNN_TENSOR_H


