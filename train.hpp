#ifndef NEURAL_TRAIN_HPP
#define NEURAL_TRAIN_HPP

namespace neural
{
    class Network;

    // Train network on mnist dataset. Assumes 784 input, 10 output.
    bool train(neural::Network& network, const char* data, const char* labels, float rate);

    void process_image(float* image, float* temp, int w, int h);
} // namespace neural

#endif // NEURAL_TRAIN_HPP