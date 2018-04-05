cmake_minimum_required(VERSION 3.9)
project(MiniCNN)

set(CMAKE_CXX_STANDARD 11)

add_executable(MiniCNN main.cpp include/Tensor.h src/Tensor.cpp include/Layer.h src/FullyConnectedLayer.cpp include/FullyConnectedLayer.h src/CalcFunctions.cpp include/CalcFunctions.h src/InputLayer.cpp include/InputLayer.h src/ActivationLayer.cpp include/ActivationLayer.h src/SoftmaxLayer.cpp include/SoftmaxLayer.h src/LossFunction.cpp include/LossFunction.h src/Optimizer.cpp include/Optimizer.h src/Network.cpp include/Network.h src/ThreadPool.cpp include/ThreadPool.h src/mnist_data_loader.cpp include/mnist_data_loader.h src/mnist_train_test.cpp include/MiniCNN.h)