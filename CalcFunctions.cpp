//
// Created by yang chen on 2018/3/7.
//
#include <cmath>
#include <random>
#include "CalcFunctions.h"
#include "Tensor.h"

namespace MiniCNN
{
    void normal_distribution_init(float* data, const unsigned int size, const float meanValue, const float standardDeviation)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::normal_distribution<float> dist(meanValue, standardDeviation);
        for (unsigned int i = 0; i < size; i++)
        {
            data[i] = dist(engine);
        }
    }

    void uniform_distribution_init(float* data, const unsigned int size, const float lowValue, const float highValue)
    {
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_real_distribution<float> dist(lowValue, highValue);
        for (unsigned int i = 0; i < size; i++)
        {
            data[i] = dist(engine);
        }
    }

    void constant_distribution_init(float* data, const unsigned int size, const float constantValue)
    {
        for (unsigned int i = 0; i < size; i++)
        {
            data[i] = constantValue;
        }
    }

    void xavier_init(float* data, const unsigned int size, const unsigned int inNum, const unsigned int outNum)
    {
        const float bias = std::sqrt(6.0f / float(inNum + outNum));
        uniform_distribution_init(data, size, -bias, bias);
    }


    float moving_average(float avg, const int acc_number, float value)
    {
        // avg = avg - avg/number + value/number
        avg -= avg / acc_number;
        avg += value / acc_number;
        return avg;
    }

    void mul(const float* a, const float* b, float* c, const unsigned int len)
    {
        for (unsigned int i = 0; i < len; i++)
        {
            c[i] = a[i]*b[i];
        }
    }

    void mul_inplace(float* a, const float* b, const unsigned int len)
    {
        for (unsigned int i = 0; i < len; i++)
        {
            a[i] *= b[i];
        }
    }

    //a /= b
    void div_inplace(float* a, const float b, const unsigned int len)
    {
        for (unsigned int i = 0; i < len;i++)
        {
            a[i] /= b;
        }
    }


    // f(x)=1/(1+e^(-x))
    static inline float sigmoid(const float x)
    {
        float result = 0;
        result = 1.0f / (1.0f + std::exp(-1.0f*x));
        return result;
    }

    void sigmoid(const float* x, float* y, const unsigned int len)
    {
        for (unsigned int i = 0; i < len; i++)
        {
            y[i] = sigmoid(x[i]);
        }
    }

    // f'(x) = x(1-x)
    static inline float df_sigmoid(const float x)
    {
        return x*(1.0f - x);
    }

    void df_sigmoid(const float* x, float* y, const unsigned int len)
    {
        for (unsigned int i = 0; i < len; i++)
        {
            y[i] = df_sigmoid(x[i]);
        }
    }


    // f(x)=max(x,0)
    static inline float relu(const float x)
    {
        float result = (x > 0.0f)? x : 0.0f;
        return result;
    }

    void relu(const float* x, float* y, const unsigned int len)
    {
        for (unsigned int i = 0; i < len; i++)
        {
            y[i] = relu(x[i]);
        }
    }

    // f'(x)=0(x<=0),1(x>0)
    static inline float df_relu(const float x)
    {
        //note : too small df is not suitable. 原因？
        return x <= 0.0f ? 0.01f : 1.0f;
    }

    void df_relu(const float* x, float* y, const unsigned int len)
    {
        for (unsigned int i = 0; i < len; i++)
        {
            y[i] = df_relu(x[i]);
        }
    }


    void fullyConnect(const float* input, const float* weight, const float* bias, float* output,
                     const unsigned int n, const unsigned int inBatchSize, const unsigned int outBatchSize)
    {

        // X' = XW + b
        // 在fully connected layer中，weights实际上是以（1，inputNum * outputNum，1，1）记录的
        // inBatchSize = 该输入层神经元数，outBatchSize = 输出层神经元数
        if (bias)
        {
            for (unsigned int k = 0; k < n; k++)
            {
                const float* pInput = input + k * inBatchSize;
                float* pOutput = output + k * outBatchSize;

                for (unsigned int i = 0; i < outBatchSize; i++)
                {
                    float sum = 0.0f;
                    for (unsigned int j = 0; j < inBatchSize; j++)
                    {
                        sum += pInput[j] * weight[i * inBatchSize + j];
                    }
                    sum += bias[i];
                    pOutput[i] = sum;
                }
            }
        }
        else
        {
            for (unsigned int k = 0; k < n; k++)
            {
                const float* pInput = input + k * inBatchSize;
                float* pOutput = output + k * outBatchSize;

                for (unsigned int i = 0; i < outBatchSize; i++)
                {
                    float sum = 0.0f;
                    for (unsigned int j = 0; j < inBatchSize; j++)
                    {
                        sum += pInput[j] * weight[i * inBatchSize + j];
                    }
                    pOutput[i] = sum;
                }
            }
        }
    }

    /*
    static void convolution2d_validate(const float* input, const float* kernel, const float* bias, float* output,
                                       const unsigned int in, const unsigned int ic, const unsigned int iw, const unsigned int ih,
                                       const unsigned int kn, const unsigned int kw, const unsigned int kh, const unsigned int kws, const unsigned int khs,
                                       const unsigned int ow, const unsigned int oh)
    {
        const Tensor inputSize(in, ic, iw, ih);
        const Tensor kernelSize(kn, ic, kw, kh);
        const Tensor outputSize(in, kn, ow, oh);
        for (unsigned int nn = 0; nn < in; nn++)
        {
            for (unsigned int nc = 0; nc < outputSize.channels; nc++)
            {
                for (unsigned int nh = 0; nh < outputSize.height; nh++)
                {
                    for (unsigned int nw = 0; nw < outputSize.width; nw++)
                    {
                        const unsigned int inStartX = nw*kws;
                        const unsigned int inStartY = nh*khs;
                        float sum = 0;
                        for (unsigned int kc = 0; kc < kernelSize.channels; kc++)
                        {
                            for (unsigned int kh = 0; kh < kernelSize.height; kh++)
                            {
                                for (unsigned int kw = 0; kw < kernelSize.width; kw++)
                                {
                                    const unsigned int prevIdx = inputSize.getIndex(nn, kc, inStartY + kh, inStartX + kw);
                                    const unsigned int kernelIdx = kernelSize.getIndex(nc, kc, kh, kw);
                                    sum += input[prevIdx] * kernel[kernelIdx];
                                }
                            }
                        }
                        if (bias)
                        {
                            const unsigned int biasIdx = nc;
                            sum += bias[biasIdx];
                        }
                        const unsigned int nextIdx = outputSize.getIndex(nn, nc, nh, nw);
                        output[nextIdx] = sum;
                    }
                }
            }
        }
    }
    static void convolution2d_same(const float* input, const float* kernel, const float* bias, float* output,
                                   const unsigned int in, const unsigned int ic, const unsigned int iw, const unsigned int ih,
                                   const unsigned int kn, const unsigned int kw, const unsigned int kh, const unsigned int kws, const unsigned int khs,
                                   const unsigned int ow, const unsigned int oh)
    {
        const Tensor inputSize(in, ic, iw, ih);
        const Tensor kernelSize(kn, ic, kw, kh);
        const Tensor outputSize(in, kn, ow, oh);
        for (unsigned int nn = 0; nn < in; nn++)
        {
            for (unsigned int nc = 0; nc < outputSize.channels; nc++)
            {
                for (unsigned int nh = 0; nh < outputSize.height; nh++)
                {
                    for (unsigned int nw = 0; nw < outputSize.width; nw++)
                    {
                        const int inStartX = nw - kw/2;
                        const int inStartY = nh - kh / 2;
                        float sum = 0;
                        for (unsigned int kc = 0; kc < kernelSize.channels; kc++)
                        {
                            for (unsigned int kh = 0; kh < kernelSize.height; kh++)
                            {
                                for (unsigned int kw = 0; kw < kernelSize.width; kw++)
                                {
                                    const int inY = inStartY + kh;
                                    const int inX = inStartX + kw;
                                    if (inY >= 0 && inY<(int)inputSize.height && inX >= 0 && inX<(int)inputSize.width)
                                    {
                                        const unsigned int prevIdx = inputSize.getIndex(nn, kc, inY, inX);
                                        const unsigned int kernelIdx = kernelSize.getIndex(nc, kc, kh, kw);
                                        sum += input[prevIdx] * kernel[kernelIdx];
                                    }
                                }
                            }
                        }
                        if (bias)
                        {
                            const unsigned int biasIdx = nc;
                            sum += bias[biasIdx];
                        }
                        const unsigned int nextIdx = outputSize.getIndex(nn, nc, nh, nw);
                        output[nextIdx] = sum;
                    }
                }
            }
        }
    }
    void convolution2d(const float* input, const float* kernel, const float* bias, float* output,
                       const unsigned int in, const unsigned int ic, const unsigned int iw, const unsigned int ih,
                       const unsigned int kn, const unsigned int kw, const unsigned int kh, const unsigned int kws, const unsigned int khs,
                       const unsigned int ow, const unsigned int oh,
                       const int mode)
    {
        if (mode == 0)
        {
            convolution2d_validate(input, kernel, bias, output, in, ic, iw, ih, kn, kw, kh, kws, khs, ow, oh);
        }else if (mode == 1)
        {
            convolution2d_same(input, kernel, bias, output, in, ic, iw, ih, kn, kw, kh, kws, khs, ow, oh);
        }
    }
     */
}//namespace