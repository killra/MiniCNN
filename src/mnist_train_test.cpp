//
// Created by yang chen on 2018/3/13.
//

#include <iostream>
#include <cassert>
#include <random>
#include "../include/MiniCNN.h"
#include "../include/mnist_data_loader.h"


const int CLASSES = 10;

static bool fetch_data(const std::vector<image_t>& images,std::shared_ptr<MiniCNN::Tensor> inputTensor,
                       const std::vector<label_t>& labels, std::shared_ptr<MiniCNN::Tensor> labelTensor,
                       const size_t offset, const size_t length)
{
    assert(images.size() == labels.size() && inputTensor->getShape().Batch == labelTensor->getShape().Batch);
    if (offset >= images.size())
    {
        return false;
    }
    size_t actualEndPos = offset + length;
    if (actualEndPos > images.size())
    {
        //image data
        auto inputDataSize = inputTensor->getShape();
        inputDataSize.Batch = images.size() - offset;
        actualEndPos = offset + inputDataSize.Batch;
        inputTensor.reset(new MiniCNN::Tensor(inputDataSize));
        //label data
        auto labelDataSize = labelTensor->getShape();
        labelDataSize.Batch = inputDataSize.Batch;
        labelTensor.reset(new MiniCNN::Tensor(inputDataSize));
    }
    //copy
    const size_t sizePerImage = inputTensor->getShape().oneBatchSize();
    const size_t sizePerLabel = labelTensor->getShape().oneBatchSize();
    assert(sizePerImage == images[0].channels*images[0].width*images[0].height);
    //scale to 0.0f~1.0f
    const float scaleRate = 1.0f / 255.0f;
    for (size_t i = offset; i < actualEndPos; i++)
    {
        //image data
        float* inputData = inputTensor->getData().get() + (i - offset)*sizePerImage;
        const uint8_t* imageData = &images[i].data[0];
        for (size_t j = 0; j < sizePerImage;j++)
        {
            inputData[j] = (float)imageData[j] * scaleRate;
        }
        //label data
        float* labelData = labelTensor->getData().get() + (i - offset)*sizePerLabel;
        const uint8_t label = labels[i].data;
        for (size_t j = 0; j < sizePerLabel; j++)
        {
            if (j == label)
            {
                labelData[j] = 1.0f;
            }
            else
            {
                labelData[j] = 0.0f;
            }
        }
    }
    return true;
}
static std::shared_ptr<MiniCNN::Tensor> convertLabelToTensor(const std::vector<label_t>& test_labels, const size_t start, const size_t len)
{
    assert(test_labels.size() > 0);
    const size_t number = len;
    const size_t sizePerLabel = CLASSES;
    std::shared_ptr<MiniCNN::Tensor> result(new MiniCNN::Tensor(MiniCNN::Shape(number, CLASSES, 1, 1)));
    for (size_t i = start; i < start + len; i++)
    {
        //image data
        float* labelData = result->getData().get() + (i - start)*sizePerLabel;
        const uint8_t label = test_labels[i].data;
        for (size_t j = 0; j < sizePerLabel; j++)
        {
            if (j == label)
            {
                labelData[j] = 1.0f;
            }
            else
            {
                labelData[j] = 0.0f;
            }
        }
    }
    return result;
}

static std::shared_ptr<MiniCNN::Tensor> convertVectorToTensor(const std::vector<image_t>& test_images, const size_t start, const size_t len)
{
    assert(test_images.size() > 0);
    const size_t number = len;
    const size_t channel = test_images[0].channels;
    const size_t width = test_images[0].width;
    const size_t height = test_images[0].height;
    const size_t sizePerImage = channel*width*height;
    const float scaleRate = 1.0f / 255.0f;
    std::shared_ptr<MiniCNN::Tensor> result(new MiniCNN::Tensor(MiniCNN::Shape(number, channel, width, height)));
    for (size_t i = start; i < start + len; i++)
    {
        //image data
        float* inputData = result->getData().get() + (i-start)*sizePerImage;
        const uint8_t* imageData = &test_images[i].data[0];
        for (size_t j = 0; j < sizePerImage; j++)
        {
            inputData[j] = (float)imageData[j] * scaleRate;
        }
    }
    return result;
}

static uint8_t getMaxIdxInArray(const float* start, const float* stop)
{
    assert(start && stop && stop >= start);
    ptrdiff_t result = 0;
    const ptrdiff_t len = stop - start;
    for (ptrdiff_t i = 0; i < len; i++)
    {
        if (start[i] >= start[result])
        {
            result = i;
        }
    }
    return (uint8_t)result;
}

static std::pair<float,float> test(MiniCNN::Network& network, const size_t batch,const std::vector<image_t>& test_images,const std::vector<label_t>& test_labels)
{
    assert(test_images.size() == test_labels.size() && test_images.size()>0);
    int correctCount = 0;
    float loss = 0.0f;
    int batchs = 0;
    for (size_t i = 0; i < test_labels.size(); i += batch, batchs++)
    {
        const size_t start = i;
        const size_t len = std::min(test_labels.size() - start, batch);
        const std::shared_ptr<MiniCNN::Tensor> inputTensor = convertVectorToTensor(test_images, start, len);
        const std::shared_ptr<MiniCNN::Tensor> labelTensor = convertLabelToTensor(test_labels, start, len);
        const std::shared_ptr<MiniCNN::Tensor> probTensor = network.testBatch(inputTensor);

        //get loss
        const float batch_loss = network.getLoss(labelTensor, probTensor);
        loss = MiniCNN::moving_average(loss, batchs + 1, batch_loss);

        const size_t labelSize = probTensor->getShape().oneBatchSize();
        const float* probData = probTensor->getData().get();
        for (size_t j = 0; j < len; j++)
        {
            const uint8_t stdProb = test_labels[i+j].data;
            const uint8_t testProb = getMaxIdxInArray(probData + j*labelSize, probData + (j + 1) * labelSize);
            if (stdProb == testProb)
            {
                correctCount++;
            }
        }
    }
    const float accuracy = (float)correctCount / (float)test_labels.size();
    return std::pair<float, float>(accuracy,loss);
}

static float getAccuracy(const std::shared_ptr<MiniCNN::Tensor> probTensor, const std::shared_ptr<MiniCNN::Tensor> labelTensor)
{
    const auto probSize = probTensor->getShape();
    const auto labelSize = labelTensor->getShape();
    const auto itemSize = labelSize.oneBatchSize();
    const float* probData = probTensor->getData().get();
    const float* labelData = labelTensor->getData().get();
    assert(probSize == labelSize);
    int correctCount = 0;
    int totalCount = 0;
    for (size_t n = 0; n < probSize.Batch;n++)
    {
        const uint8_t stdProb = getMaxIdxInArray(labelData + n*itemSize, labelData + (n + 1) * itemSize);
        const uint8_t testProb = getMaxIdxInArray(probData + n*itemSize, probData + (n + 1) * itemSize);
        if (stdProb == testProb)
        {
            correctCount++;
        }
        totalCount++;
    }
    const float result = (float)correctCount / (float)totalCount;
    return result;
}

static void add_input_layer(MiniCNN::Network& network)
{
    std::shared_ptr<MiniCNN::InputLayer> inputLayer(std::make_shared<MiniCNN::InputLayer>());
    network.addLayer(inputLayer);
}

static void add_fc_layer(MiniCNN::Network& network, const int output_count)
{
    std::shared_ptr<MiniCNN::FullyConnectedLayer> fullconnectLayer(std::make_shared<MiniCNN::FullyConnectedLayer>());
    fullconnectLayer->setParameters(MiniCNN::Shape(1, output_count, 1, 1), true);
    network.addLayer(fullconnectLayer);
}

static void add_active_layer(MiniCNN::Network& network)
{
    network.addLayer(std::make_shared<MiniCNN::ReluLayer>());
}

static void add_softmax_layer(MiniCNN::Network& network)
{
    std::shared_ptr<MiniCNN::SoftmaxLayer> softmaxLayer(std::make_shared<MiniCNN::SoftmaxLayer>());
    network.addLayer(softmaxLayer);
}

static void shuffle_data(std::vector<image_t>& images, std::vector<label_t>& labels)
{
    assert(images.size() == labels.size());
    std::vector<size_t> indexArray;
    for (size_t i = 0; i < images.size();i++)
    {
        indexArray.push_back(i);
    }
    std::random_shuffle(indexArray.begin(), indexArray.end());

    std::vector<image_t> tmpImages(images.size());
    std::vector<label_t> tmpLabels(labels.size());
    for (size_t i = 0; i < images.size(); i++)
    {
        const size_t srcIndex = i;
        const size_t dstIndex = indexArray[i];
        tmpImages[srcIndex] = images[dstIndex];
        tmpLabels[srcIndex] = labels[dstIndex];
    }
    images = tmpImages;
    labels = tmpLabels;
}

/***************************  not finished **************************************
static void add_conv_layer(MiniCNN::Network& network,const int number,const int input_channel)
{
    std::shared_ptr<MiniCNN::ConvolutionLayer> convLayer(std::make_shared<MiniCNN::ConvolutionLayer>());
    convLayer->setParamaters(MiniCNN::ParamSize(number, input_channel, 3, 3), 1, 1, true, MiniCNN::ConvolutionLayer::SAME);
    network.addLayer(convLayer);
}

static void add_pool_layer(MiniCNN::Network& network, const int number)
{
    std::shared_ptr<MiniCNN::PoolingLayer> poolingLayer(std::make_shared<MiniCNN::PoolingLayer>());
    poolingLayer->setParamaters(MiniCNN::PoolingLayer::PoolingType::MaxPooling, MiniCNN::ParamSize(1, number, 2, 2), 2, 2, MiniCNN::PoolingLayer::SAME);
    network.addLayer(poolingLayer);
}

static MiniCNN::Network buildConvNet(const size_t batch,const size_t channels,const size_t width,const size_t height)
{
    MiniCNN::Network network;
    network.setInputSize(MiniCNN::Shape(batch, channels, width, height));

    //input data layer
    add_input_layer(network);

    //convolution layer
    add_conv_layer(network, 6 ,1);
    add_active_layer(network);
    //pooling layer
    add_pool_layer(network, 6);

    //convolution layer
    add_conv_layer(network, 12, 6);
    add_active_layer(network);
    //pooling layer
    add_pool_layer(network, 12);

    //full connect layer
    add_fc_layer(network, 512);
    add_active_layer(network);

    //network.addLayer(std::make_shared<MiniCNN::DropoutLayer>(0.5f));

    //full connect layer
    add_fc_layer(network, CLASSES);

    //soft max layer
    add_softmax_layer(network);

    return network;
}
*****************************************************************************/


static MiniCNN::Network buildMLPNet(const size_t batch, const size_t channels, const size_t width, const size_t height)
{
    MiniCNN::Network network;
    network.setInputSize(MiniCNN::Shape(batch, channels, width, height));

    //input data layer
    add_input_layer(network);

    //full connect layer
    add_fc_layer(network, 512);
    add_active_layer(network);

    //full connect layer
    add_fc_layer(network, 256);
    add_active_layer(network);

    //full connect layer
    add_fc_layer(network, CLASSES);

    //soft max layer
    add_softmax_layer(network);

    return network;
}

static void train(const std::string& mnist_train_images_file,
                  const std::string& mnist_train_labels_file,
                  const std::string& modelFilePath)
{
    bool success = false;

    //load train images
    std::cout <<"loading training data..." << std::endl;

    std::vector<image_t> images;
    success = load_mnist_images(mnist_train_images_file, images);
    assert(success && images.size() > 0);
    //load train labels
    std::vector<label_t> labels;
    success = load_mnist_labels(mnist_train_labels_file, labels);
    assert(success && labels.size() > 0);
    assert(images.size() == labels.size());
    shuffle_data(images, labels);

    //train data & validate data
    //train
    std::vector<image_t> train_images(static_cast<size_t>(images.size()*0.9f));
    std::vector<label_t> train_labels(static_cast<size_t>(labels.size()*0.9f));
    std::copy(images.begin(), images.begin() + train_images.size(), train_images.begin());
    std::copy(labels.begin(), labels.begin() + train_labels.size(), train_labels.begin());
    //validate
    std::vector<image_t> validate_images(images.size() - train_images.size());
    std::vector<label_t> validate_labels(labels.size() - train_labels.size());
    std::copy(images.begin() + train_images.size(), images.end(), validate_images.begin());
    std::copy(labels.begin() + train_labels.size(), labels.end(), validate_labels.begin());

    std::cout << "load training data done. train set's size is " << train_images.size()
              << ", validate set's size is " << validate_images.size() << std::endl;

    float learningRate = 0.1f;
    const float decayRate = 0.8f;
    const float minLearningRate = 0.001f;
    const unsigned int testAfterBatches = 10;
    const unsigned int maxBatches = 10000;
    const unsigned int max_epoch = 5;
    const unsigned int batch = 128;
    const unsigned int channels = images[0].channels;
    const unsigned int width = images[0].width;
    const unsigned int height = images[0].height;

    printf("max_epoch:%d, testAfterBatches:%d \n", max_epoch, testAfterBatches);
    printf("learningRate:%f, decayRate:%f, minLearningRate:%f \n", learningRate, decayRate, minLearningRate);
    printf("channels:%d, width:%d, height:%d \n", channels, width, height);

    std::cout << "construct network begin..." << std::endl;

    MiniCNN::Network network(buildMLPNet(batch, channels, width, height));
    network.setLossFunction(std::make_shared<MiniCNN::CrossEntropyFunction>());
    network.setOptimizer(std::make_shared<MiniCNN::SGD>(learningRate));
    network.setLearningRate(learningRate);

    std::cout << "construct network done." << std::endl;

    float val_accuracy = 0.0f;
    float train_loss = 0.0f;
    int train_batches = 0;
    float val_loss = 0.0f;

    //train
    std::cout << "begin training..." << std::endl;
    std::shared_ptr<MiniCNN::Tensor> inputTensor = std::make_shared<MiniCNN::Tensor>(MiniCNN::Shape(batch, channels, width, height));
    std::shared_ptr<MiniCNN::Tensor> labelTensor = std::make_shared<MiniCNN::Tensor>(MiniCNN::Shape(batch, CLASSES, 1, 1));
    unsigned int epochIdx = 0;
    while (epochIdx < max_epoch)
    {
        //before epoch start, shuffle all train data first
        shuffle_data(train_images, train_labels);
        unsigned int batchIdx = 0;
        while (true)
        {
            if (!fetch_data(train_images, inputTensor, train_labels, labelTensor, batchIdx*batch, batch))
            {
                break;
            }
            const float batch_loss = network.trainBatch(inputTensor,labelTensor);
            train_loss = MiniCNN::moving_average(train_loss, train_batches + 1, batch_loss);
            train_batches++;

            if (batchIdx > 0 && batchIdx % testAfterBatches == 0)
            {
                std::tie(val_accuracy, val_loss) = test(network, 128, validate_images, validate_labels);

                printf("sample:%d/%lu, learningRate:%f, train_loss:%f, val_loss:%f, val_accuracy:%.4f%% \n",
                                     batchIdx*batch, train_images.size(), learningRate, train_loss, val_loss, val_accuracy*100.0f);

                train_loss = 0.0f;
                train_batches = 0;
            }
            if (batchIdx >= maxBatches)
            {
                break;
            }
            batchIdx++;
        }
        if (batchIdx >= maxBatches)
        {
            break;
        }

        std::tie(val_accuracy, val_loss) = test(network, 128, validate_images, validate_labels);

        //update learning rate
        learningRate = std::max(learningRate*decayRate, minLearningRate);
        network.setLearningRate(learningRate);

        printf("epoch[%d] val_loss : %f , val_accuracy : %.4f%%  \n", epochIdx++, val_loss, val_accuracy*100.0f);
    }

    std::tie(val_accuracy, val_loss) = test(network, 128, validate_images, validate_labels);
    printf("final val_loss : %f , final val_accuracy : %.4f%% \n", val_loss, val_accuracy*100.0f);

    success = network.saveModel(modelFilePath);
    assert(success);

    std::cout << "finished training." << std::endl;
}

static void test(const std::string& mnist_test_images_file,
                 const std::string& mnist_test_labels_file,
                 const std::string& modelFilePath)
{
    bool success = false;

    //load train images
    printf("loading test data...\n");

    std::vector<image_t> images;
    success = load_mnist_images(mnist_test_images_file, images);
    assert(success && images.size() > 0);

    //load train labels
    std::vector<label_t> labels;
    success = load_mnist_labels(mnist_test_labels_file, labels);
    assert(success && labels.size() > 0);
    assert(images.size() == labels.size());
    printf("load test data done. images' size is %lu,validate labels' size is %lu \n", images.size(), labels.size());

    const unsigned int batch = 64;
    const unsigned int channels = images[0].channels;
    const unsigned int width = images[0].width;
    const unsigned int height = images[0].height;
    printf("channels:%d , width:%d , height:%d \n", channels, width, height);

    printf("construct network begin...\n");
    MiniCNN::Network network;
    success = network.loadModel(modelFilePath);
    assert(success);
    printf("construct network done.\n");

    //train
    printf("begin test...\n");
    float accuracy = 0.0f, loss = std::numeric_limits<float>::max();
    std::tie(accuracy,loss) = test(network,batch,images, labels);
    printf("accuracy : %.4f%% \n", accuracy*100.0f);
    printf("finished test. \n");
}

int mnist_main()
{
    MiniCNN::set_thread_num(4);

    const std::string model_file = "../model/mnist.modelx";

    const std::string mnist_train_images_file = "../res/MNIST_data/train-images-idx3-ubyte";
    const std::string mnist_train_labels_file = "../res/MNIST_data/train-labels-idx1-ubyte";
    train(mnist_train_images_file, mnist_train_labels_file, model_file);
    system("pause");

    //NOTE : NEVER NEVER fine tune network for the test accuracy!!!
    const std::string mnist_test_images_file = "../res/MNIST_data/t10k-images-idx3-ubyte";
    const std::string mnist_test_labels_file = "../res/MNIST_data/t10k-labels-idx1-ubyte";
    test(mnist_test_images_file, mnist_test_labels_file, model_file);

    return 0;
}