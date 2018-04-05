//
// Created by yang chen on 2018/3/13.
//

#ifndef MINICNN_MNIST_DATA_LOADER_H
#define MINICNN_MNIST_DATA_LOADER_H

#include <vector>
#include <cstdint>

struct image_t
{
    unsigned int width, height, channels;
    std::vector<uint8_t> data;
};
bool load_mnist_images(const std::string& file_path, std::vector<image_t>& images);

struct label_t
{
    uint8_t data;
};
bool load_mnist_labels(const std::string& file_path, std::vector<label_t>& labels);


#endif //MINICNN_MNIST_DATA_LOADER_H
