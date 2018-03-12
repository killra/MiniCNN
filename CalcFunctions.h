//
// Created by yang chen on 2018/3/7.
//

#ifndef MINICNN_CALCFUNCTIONS_H
#define MINICNN_CALCFUNCTIONS_H



namespace MiniCNN
{
    void normal_distribution_init(float* data, const unsigned int size, const float meanValue, const float standardDeviation);
    void uniform_distribution_init(float* data, const unsigned int size, const float lowValue, const float highValue);
    void constant_distribution_init(float* data, const unsigned int size, const float constantValue);
    void xavier_init(float* data, const unsigned int size, const unsigned int inNum, const unsigned int outNum);

    float moving_average(float avg, const int acc_number, float value);

    // c = a*b
    void mul(const float* a, const float* b, float* c, const unsigned int len);
    // a *= b
    void mul_inplace(float* a, const float* b, const unsigned int len);
    // a /= b
    void div_inplace(float* a, const float b, const unsigned int len);

    void sigmoid(const float* x, float* y, const unsigned int len);
    void df_sigmoid(const float* x, float* y, const unsigned int len);

    void relu(const float* x, float* y, const unsigned int len);
    void df_relu(const float* x, float* y, const unsigned int len);

    // 待更新，参数重命名
    void fullyConnect(const float* input, const float* weight, const float* bias,float* output,
                     const unsigned int n, const unsigned int inBatchSize, const unsigned int outBatchSize);

    // mode: 0-valid,1-same
    // 待更新，参数太多了
    void convolution2d(const float* input, const float* kernel, const float* bias, float* output,
                       const unsigned int in, const unsigned int ic, const unsigned int iw, const unsigned int ih,
                       const unsigned int kn, const unsigned int kw, const unsigned int kh, const unsigned int kws, const unsigned int khs,
                       const unsigned int ow, const unsigned int oh,
                       const int mode);
};

#endif //MINICNN_CALCFUNCTIONS_H