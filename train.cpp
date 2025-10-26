#include "train.hpp"
#include "network.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace neural
{
    bool is_little_endian() {
        uint16_t x = 1;
        return *reinterpret_cast<uint8_t*>(&x) == 1;
    }

    uint32_t byte_swap(uint32_t val) {
        return ((val & 0x000000FF) << 24) |
               ((val & 0x0000FF00) << 8)  |
               ((val & 0x00FF0000) >> 8)  |
               ((val & 0xFF000000) >> 24);
    }

    bool train(neural::Network& network, const char* data, const char* labels, float rate) {
        FILE* fdata = fopen(data, "rb");
        FILE* flabels = fopen(labels, "rb");

        if (!fdata || !flabels)
            return false;

        uint32_t magic, num, rows, cols;
        fread(&magic, sizeof(magic), 1, fdata);
        fread(&num, sizeof(num), 1, fdata);
        fread(&rows, sizeof(rows), 1, fdata);
        fread(&cols, sizeof(cols), 1, fdata);

        uint32_t n;
        fread(&magic, sizeof(magic), 1, flabels);
        fread(&n, sizeof(n), 1, flabels);

        if (is_little_endian()) {
            num = byte_swap(num);
            rows = byte_swap(rows);
            cols = byte_swap(cols);
            n = byte_swap(n);
        }

        if (n != num || network.input() != rows * cols || network.output() != 10)
            return false;

        uint8_t* label = new uint8_t[n];
        fread(label, 1, n, flabels);
        fclose(flabels);

        uint8_t* image = new uint8_t[rows * cols];
        float* fimage = new float[rows * cols * 2];

        float output[10];
        for (int i = 0; i < num; ++i) {
            std::cout << "\r" << (i + 1) << "/" << num << std::flush;

            fread(image, rows * cols, 1, fdata);

            for (int j = 0; j < rows * cols; ++j) {
                fimage[j] = static_cast<float>(image[j]) / 255.0f;
            }

            network.compute(fimage, nullptr);

            for (int k = 0; k < 10; ++k) {
                output[k] = k == static_cast<int>(label[i]) ? 0.9f : 0.1f;
            }

            network.backpropagate(output, rate);
        }

        fclose(fdata);

        delete[] label;
        delete[] image;
        delete[] fimage;

        std::cout << std::endl;

        return true;
    }
} // namespace neural
