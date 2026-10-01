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

            process_image(fimage, fimage + rows * cols, cols, rows);

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
    static float pixel(float* image, int w, int h, int x, int y) {
        if (x < 0 || y < 0 || x >= w || y >= h)
            return 0.0f;

        return image[y * w + x];
    }

    // Bilinear interpolation of the four pixels around (x, y).
    static float sample(float* image, int w, int h, float x, float y) {
        int left = static_cast<int>(std::floor(x));
        int top = static_cast<int>(std::floor(y));
        float tx = x - left;
        float ty = y - top;

        float top_left = pixel(image, w, h, left, top);
        float top_right = pixel(image, w, h, left + 1, top);
        float bottom_left = pixel(image, w, h, left, top + 1);
        float bottom_right = pixel(image, w, h, left + 1, top + 1);

        float upper = top_left + tx * (top_right - top_left);
        float lower = bottom_left + tx * (bottom_right - bottom_left);

        return upper + ty * (lower - upper);
    }

    void process_image(float* image, float* temp, int w, int h) {
        // Copy into temp while finding the bounding box and center of mass.
        int left = w, right = -1, top = h, bottom = -1;
        float mass = 0.0f, center_x = 0.0f, center_y = 0.0f;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float value = image[y * w + x];
                temp[y * w + x] = value;
                if (value <= 0.0f)
                    continue;

                left = std::min(left, x);
                right = std::max(right, x);
                top = std::min(top, y);
                bottom = std::max(bottom, y);

                mass += value;
                center_x += value * x;
                center_y += value * y;
            }
        }

        if (mass <= 0.0f)
            return;

        center_x /= mass;
        center_y /= mass;

        // Drawing pixels per output pixel, so the longer side of the box becomes BOX.
        float scale = static_cast<float>(std::max(right - left, bottom - top) + 1) / BOX;

        // Resample around the center of mass so it lands in the middle of the image.
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float source_x = center_x + (x - w / 2.0f) * scale;
                float source_y = center_y + (y - h / 2.0f) * scale;
                image[y * w + x] = sample(temp, w, h, source_x, source_y);
            }
        }
    }
} // namespace neural