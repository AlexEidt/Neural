#include "network.hpp"
#include <cmath>
#include <cfloat>

namespace neural
{
    Network::Network(int input, int* hidden, int n_hidden, int output) {
        {
            this->n_input = input;
            this->n_output = output;
            this->n_layers = 1 + n_hidden + 1;
            this->layers = new Network::Layer[this->n_layers];
        }

        {
            this->n_neurons = input + output;
            for (int i = 0; i < n_hidden; ++i) {
                this->n_neurons += hidden[i];
            }

            this->neurons = new Network::Neuron[this->n_neurons];
        }

        {
            this->n_weights = input * hidden[0];

            for (int i = 1; i < n_hidden; ++i) {
                this->n_weights += hidden[i - 1] * hidden[i];
            }

            this->n_weights += hidden[n_hidden - 1] * output;
            this->weights = new float[this->n_weights];
        }

        // Connect layers.
        {
            // next/prev
            for (int i = 0; i < this->n_layers - 1; ++i) {
                this->layers[i].next = &this->layers[i + 1];
            }

            for (int i = 1; i < this->n_layers; ++i) {
                this->layers[i].prev = &this->layers[i - 1];
            }

            // size
            this->layers[0].size = input;
            this->layers[this->n_layers - 1].size = output;

            for (int i = 0; i < n_hidden; ++i) {
                this->layers[i + 1].size = hidden[i];
            }

            // neurons
            Network::Neuron* nptr = this->neurons;
            float* fptr = this->weights;
            for (int i = 0; i < this->n_layers; ++i) {
                Network::Layer& current = this->layers[i];
                current.neurons = nptr;
                nptr += current.size;

                for (int j = 0; i > 0 && j < current.size; ++j) {
                    Network::Layer& prev = this->layers[i - 1];
                    Network::Neuron& neuron = current.neurons[j];

                    neuron.weights = fptr;
                    fptr += prev.size;
                }
            }
        }

        // Seeding.
        {
            for (int i = 0; i < this->n_weights; ++i) {
                this->weights[i] = (static_cast<float>(rand()) / RAND_MAX) * 0.2f - 0.1f;
            }
        }
    }

    Network::~Network() {
        delete[] this->weights;
        delete[] this->neurons;
        delete[] this->layers;
    }

    int Network::input() {
        return this->n_input;
    }

    int Network::output() {
        return this->n_output;
    }

    void Network::compute(float* input, float* output) {
        Network::Layer& in = this->layers[0];
        for (int i = 0; i < in.size; ++i) {
            in.neurons[i].value = input[i];
        }

        for (int i = 1; i < this->n_layers; ++i) {
            Network::Layer& prev = this->layers[i - 1];
            Network::Layer& layer = this->layers[i];

            for (int j = 0; j < layer.size; ++j) {
                Network::Neuron& neuron = layer.neurons[j];

                float total = neuron.bias;
                for (int k = 0; k < prev.size; ++k) {
                    total += prev.neurons[k].value * neuron.weights[k];
                }

                // Sigmoid activation.
                neuron.value = 1.0f / (1.0f + expf(-total));
            }
        }

        Network::Layer& out = this->layers[this->n_layers - 1];
        for (int i = 0; output != nullptr && i < out.size; ++i) {
            output[i] = out.neurons[i].value;
        }
    }

    void Network::backpropagate(float* expected, float rate) {
        Network::Layer& out = this->layers[this->n_layers - 1];
        for (int i = 0; i < out.size; ++i) {
            Network::Neuron& neuron = out.neurons[i];
            neuron.error = expected[i] - neuron.value;
        }

        // Backpropagate through all hidden layers.
        for (int i = this->n_layers - 2; i > 0; --i) {
            Network::Layer& layer = this->layers[i];
            // Sum partial derivative rates of all previous errors.
            for (int j = 0; j < layer.size; ++j) {
                Network::Neuron& neuron = layer.neurons[j];

                float sum = 0.0f;
                for (int k = 0; k < layer.next->size; ++k) {
                    Network::Neuron& output = layer.next->neurons[k];
                    sum += output.error * output.weights[j];
                }

                neuron.error = sum * neuron.value * (1.0f - neuron.value);
            }
        }

        // Update weights only after all errors are computed so each layer sees the weights from the forward pass.
        for (int i = 1; i < this->n_layers; ++i) {
            Network::Layer& layer = this->layers[i];
            for (int j = 0; j < layer.size; ++j) {
                gradient_descent(layer, layer.neurons[j], rate);
            }
        }
    }

    void Network::gradient_descent(Network::Layer& layer, Network::Neuron& neuron, float rate) {
        for (int i = 0; i < layer.prev->size; ++i) {
            Network::Neuron& prev = layer.prev->neurons[i];

            float gradient = -prev.value * neuron.error;
            // Stochastic Gradient Descent.
            neuron.weights[i] -= rate * gradient;
        }

        neuron.bias += rate * neuron.error;
    }

    int Network::argmax(float* arr, int size) {
        int index = -1;
        float max = -FLT_MAX;
        for (int i = 0; i < size; ++i) {
            if (arr[i] > max) {
                max = arr[i];
                index = i;
            }
        }

        return index;
    }
} // namespace neural