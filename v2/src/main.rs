use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::{error::Error, fs::File};
use csv::ReaderBuilder;
use rand::seq::SliceRandom;

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
    fn new(input_size: usize, layer1_size: usize, layer2_size: usize, output_size: usize, learning_rate: f32) -> Self {
        // standard xavier/glorot initialization
        // scale calculated as sqrt(6 / (fan_in + fan_out))
        let scale1 = (6.0 / (input_size + layer1_size) as f32).sqrt();
        let scale2 = (6.0 / (layer1_size + layer2_size) as f32).sqrt();
        let scale3 = (6.0 / (layer2_size + output_size) as f32).sqrt();
        
        let weights_input_to_layer1 = Array::random(
            (input_size, layer1_size), 
            Uniform::new(-scale1, scale1)
        );
        let weights_layer1_to_layer2 = Array::random(
            (layer1_size, layer2_size), 
            Uniform::new(-scale2, scale2)
        );
        let weights_layer2_to_output = Array::random(
            (layer2_size, output_size), 
            Uniform::new(-scale3, scale3)
        );

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

    // sigmoid activation function
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    // derivative of sigmoid for backpropagation
    fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|val| val * (1.0 - val))
    }

    // forward propagation
    fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // first layer
        let layer1_weighted_sum = input.dot(&self.weights_input_to_layer1);
        let layer1_activation = Self::sigmoid(&layer1_weighted_sum);
        
        // second layer
        let layer2_weighted_sum = layer1_activation.dot(&self.weights_layer1_to_layer2);
        let layer2_activation = Self::sigmoid(&layer2_weighted_sum);
        
        // output layer
        let output_weighted_sum = layer2_activation.dot(&self.weights_layer2_to_output);
        let output_activation = Self::sigmoid(&output_weighted_sum);
        
        (layer1_activation, layer2_activation, output_activation)
    }

    // backpropagation
    fn backward(
        &mut self,
        input: &Array2<f32>,
        layer1_activation: &Array2<f32>,
        layer2_activation: &Array2<f32>,
        output_activation: &Array2<f32>,
        target: &Array2<f32>,
    ) {
        // calculate error at output
        let output_error = target - output_activation;
        
        // calculate gradient for output layer
        let output_gradient = &output_error * &Self::sigmoid_derivative(output_activation);
        
        // more conservative learning rate adjustments for each layer
        // output layer
        let output_weight_adjustment = layer2_activation.t().dot(&output_gradient) * (self.learning_rate * 1.1);
        self.weights_layer2_to_output = &self.weights_layer2_to_output + &output_weight_adjustment;
        
        // calculate error and gradient for layer 2
        let layer2_error = output_gradient.dot(&self.weights_layer2_to_output.t());
        let layer2_gradient = &layer2_error * &Self::sigmoid_derivative(layer2_activation);
        
        // middle layer
        let layer2_weight_adjustment = layer1_activation.t().dot(&layer2_gradient) * (self.learning_rate * 1.05);
        self.weights_layer1_to_layer2 = &self.weights_layer1_to_layer2 + &layer2_weight_adjustment;
        
        // calculate error and gradient for layer 1
        let layer1_error = layer2_gradient.dot(&self.weights_layer1_to_layer2.t());
        let layer1_gradient = &layer1_error * &Self::sigmoid_derivative(layer1_activation);
        
        // input layer - use base learning rate
        let layer1_weight_adjustment = input.t().dot(&layer1_gradient) * self.learning_rate;
        self.weights_input_to_layer1 = &self.weights_input_to_layer1 + &layer1_weight_adjustment;
    }
    
    // train the network with a single example
    fn train(&mut self, input: &Array2<f32>, target: &Array2<f32>) {
        let (layer1_activation, layer2_activation, output_activation) = self.forward(input);
        self.backward(input, &layer1_activation, &layer2_activation, &output_activation, target);
    }
    
    // predict the digit for a given input
    fn predict(&self, input: &Array2<f32>) -> usize {
        let (_, _, output_activation) = self.forward(input);
        
        // find the index of the maximum value in the output
        let mut max_index = 0;
        let mut max_value = output_activation[[0, 0]];
        
        for i in 1..self.output_size {
            if output_activation[[0, i]] > max_value {
                max_value = output_activation[[0, i]];
                max_index = i;
            }
        }
        
        max_index
    }
}

fn load_mnist_data(filename: &str) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>), Box<dyn Error>> {
    let file = File::open(filename)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(file);
    
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    // read each row
    for result in reader.records() {
        let record = result?;
        
        // first value is the label
        let label: u8 = record[0].parse()?;
        
        // rest of the values are pixels
        let mut normalized_pixels = Vec::with_capacity(784);
        for i in 1..record.len() {
            let pixel: u8 = record[i].parse()?;
            
            // standard normalization to [0.01, 0.99] range
            // avoids saturation at sigmoid extremes
            let normalized_value = 0.01 + (pixel as f32 / 255.0) * 0.98;
            normalized_pixels.push(normalized_value);
        }
        
        // create a 1x784 array for the input
        let input = Array::from_shape_vec((1, 784), normalized_pixels)?;
        inputs.push(input);
        
        // create a one-hot encoded target (1x10 array)
        let mut target = Array::zeros((1, 10));
        target[[0, label as usize]] = 1.0;
        targets.push(target);
    }
    
    Ok((inputs, targets))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Loading training data...");
    let (train_inputs, train_targets) = load_mnist_data("../data/mnist_train.csv")?;
    
    println!("Loading test data...");
    let (test_inputs, test_targets) = load_mnist_data("../data/mnist_test.csv")?;
    
    // optimized architecture based on proven mnist results
    // 784 neurons for input (one per pixel)
    // 300 neurons in first hidden layer - good balance
    // 100 neurons in second hidden layer - effective for digits
    // 10 neurons for output (one per digit)
    // initial learning rate of 0.1 for stable convergence
    let mut nn = NeuralNetwork::new(784, 300, 100, 10, 0.1);
    
    // train the network for 3 epochs
    println!("Training the neural network...");
    let epochs = 3;
    
    for epoch in 0..epochs {
        println!("Epoch {} of {}", epoch + 1, epochs);
        
        // shuffle the training data for each epoch
        let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        // more conservative learning rate schedule for stable learning
        if epoch == 1 {
            nn.learning_rate = 0.25; // second epoch //increased learning rate from 0.05 to 0.1
        } else if epoch == 2 {
            nn.learning_rate = 0.15; // final epoch
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            nn.train(&train_inputs[idx], &train_targets[idx]);
            
            if (i + 1) % 1000 == 0 {
                println!("  Processed {} of {} training examples", i + 1, train_inputs.len());
            }
        }
        
        // each sample tested after each run
        let mut correct = 0;
        let sample_size = 2000; // larger sample for better accuracy
        for (idx, _) in indices.iter().enumerate().take(sample_size) {
            let predicted = nn.predict(&train_inputs[idx]);
            let actual = train_targets[idx].iter().position(|&x| x == 1.0).unwrap_or(0);
            if predicted == actual {
                correct += 1;
            }
        }
        let train_accuracy = (correct as f32) / (sample_size as f32) * 100.0;
        println!("  Training accuracy after epoch {}: {:.2}%", epoch + 1, train_accuracy);
        println!("  Learning rate: {}", nn.learning_rate);
    }
    
    // test the network
    println!("Testing the neural network...");
    let mut correct_predictions = 0;
    
    for (i, (input, target)) in test_inputs.iter().zip(test_targets.iter()).enumerate() {
        let predicted = nn.predict(input);
        let actual = target.iter().position(|&x| x == 1.0).unwrap_or(0);
        
        if predicted == actual {
            correct_predictions += 1;
        }
        
        if (i + 1) % 1000 == 0 {
            println!("  Processed {} of {} test examples", i + 1, test_inputs.len());
            // show intermediate accuracy
            let current_accuracy = (correct_predictions as f32) / (i as f32 + 1.0) * 100.0;
            println!("  Current accuracy: {:.2}%", current_accuracy);
        }
    }
    
    let accuracy = (correct_predictions as f32) / (test_inputs.len() as f32) * 100.0;
    println!("Final Accuracy: {:.2}%", accuracy);
    
    Ok(())
} 