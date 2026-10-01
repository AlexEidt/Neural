#include "train.hpp"
#include "network.hpp"
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <iostream>

#define BOX 20 // MNIST digits are resized so their longer side is this many pixels.

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

            process_image(fimage, fimage + rows * cols, rows, cols);

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

    // Pixel value, or 0 outside the image.
    static float pixel(float* image, int rows, int cols, int x, int y) {
        if (x < 0 || y < 0 || x >= cols || y >= rows)
            return 0.0f;

        return image[y * cols + x];
    }

    void process_image(float* image, float* temp, int rows, int cols) {
        // Bounding box of the digit.
        int left = cols, right = -1, top = rows, bottom = -1;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                if (image[y * cols + x] > 0.0f) {
                    left = std::min(left, x);
                    right = std::max(right, x);
                    top = std::min(top, y);
                    bottom = std::max(bottom, y);
                }
            }
        }

        if (right < 0)
            return;

        // Resize the box so its longer side is BOX pixels, into the top left of temp.
        float scale = static_cast<float>(std::max(right - left + 1, bottom - top + 1)) / BOX;
        float mass = 0.0f, mass_x = 0.0f, mass_y = 0.0f;
        for (int i = 0; i < rows * cols; ++i) {
            temp[i] = 0.0f;
        }

        for (int y = 0; y < BOX; ++y) {
            for (int x = 0; x < BOX; ++x) {
                // Bilinear sample of the source at this pixel's center.
                float sx = left + (x + 0.5f) * scale - 0.5f;
                float sy = top + (y + 0.5f) * scale - 0.5f;
                int x0 = static_cast<int>(std::floor(sx));
                int y0 = static_cast<int>(std::floor(sy));
                float tx = sx - x0;
                float ty = sy - y0;

                float value = (1.0f - ty) * ((1.0f - tx) * pixel(image, rows, cols, x0, y0) + tx * pixel(image, rows, cols, x0 + 1, y0)) +
                              ty * ((1.0f - tx) * pixel(image, rows, cols, x0, y0 + 1) + tx * pixel(image, rows, cols, x0 + 1, y0 + 1));

                temp[y * cols + x] = value;
                mass += value;
                mass_x += value * x;
                mass_y += value * y;
            }
        }

        if (mass <= 0.0f)
            return;

        // Move it so its center of mass is in the middle of the image.
        int shift_x = static_cast<int>(std::lround(cols / 2.0f - mass_x / mass));
        int shift_y = static_cast<int>(std::lround(rows / 2.0f - mass_y / mass));
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                image[y * cols + x] = pixel(temp, rows, cols, x - shift_x, y - shift_y);
            }
        }
    }
} // namespace neural