#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

namespace neural {
    class Network {
        public:
            Network(int input, int* hidden, int n_hidden, int output);
            ~Network();

            int input();
            int output();

            void compute(float* input, float* output);
            void backpropagate(float* expected, float rate);

            // Dump/restore the layer layout and the weights array, which holds everything the network learned.
            bool save(const char* path);
            bool load(const char* path);

            static int argmax(float* arr, int size);

        private:
            struct Layer; // Forward declaration.

            struct Neuron {
                float* weights {nullptr}; // One per previous neuron, bias last.

                float value;
                float error;
            };

            struct Layer {
                Neuron* neurons {nullptr};
                int size {0};

                Layer* prev {nullptr};
                Layer* next {nullptr};
            };

            int n_input, n_output;
            int n_layers;
            Layer* layers;

            long n_neurons;
            Neuron* neurons;

            long n_weights;
            float* weights;

            void gradient_descent(Network::Layer& layer, Network::Neuron& neuron, float rate);
    };
}

#endif // NEURAL_NETWORK_HPP