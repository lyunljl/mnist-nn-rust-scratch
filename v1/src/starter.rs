use ndarray::{s, Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::{error::Error, fs::File};

struct NeuralNetwork {
    input_size: usize,
    layer1_size: usize,
    layer2_size: usize,
    output_size: usize,
    learning_rate: f32,
    weights_input_to_layer1: Array2<f32>,
    weights_layer1_to_layer2: Array2<f32>,
    weights_layer2_to_output: Array2<f32>,
}

impl NeuralNetwork {
    fn new(input_size: usize, layer1_size: usize, layer2_size:usize, output_size: usize, learning_rate: f32) -> Self {
        // Initialize the weights for the input and hidden layers randomly between 0 and 0.1.
        let weights_input_to_layer1 = Array::random((input_size, layer1_size), Uniform::new(0.0, 0.1));
        let weights_layer1_to_layer2 = Array::random((layer1_size, layer2_size), Uniform::new(0.0, 0.1));
        let weights_layer2_to_output =
            Array::random((layer2_size, output_size), Uniform::new(0.0, 0.1));

        // Return a neural network that has the randomly initialized weights.
        NeuralNetwork {
            input_size,
            layer1_size,
            layer2_size,
            output_size,
            learning_rate,
            weights_input_to_layer1,
            weights_layer1_to_layer2,
            weights_layer2_to_output,
        }
    }

    // Forward propagation.  Returns all of the intermediate and final outputs.
    // Currently commented out to avoid compilations errors
    fn forward(&self, input: &Array2<f32>) // -> (Array2<f32>, Array2<f32>, Array2<f32>) 
    {
        // (layer1_output, layer2_output, final_output)
    }

    // Backpropagation pass through the network. No return values but it is supposed to
    // update all weights.  It accepts the input, intermediate outputs and final outputs
    // as parameters as well as the target values.
    fn backward(
        &mut self,
        input: &Array2<f32>,
        layer1_output: &Array2<f32>,
        layer2_output: &Array2<f32>,
        final_output: &Array2<f32>,
        target: &Array2<f32>,
    ) {
    }
}

fn main() {
}
