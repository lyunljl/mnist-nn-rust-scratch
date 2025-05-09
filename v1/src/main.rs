use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::{error::Error, fs::File};
use csv::ReaderBuilder;
use rand::thread_rng;
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
    momentum_input_to_layer1: Array2<f32>,
    momentum_layer1_to_layer2: Array2<f32>,
    momentum_layer2_to_output: Array2<f32>,
    momentum_factor: f32,
}

impl NeuralNetwork {
    fn new(input_size: usize, layer1_size: usize, layer2_size:usize, output_size: usize, learning_rate: f32) -> Self {
        // Initialize weights with a better range for Xavier/Glorot initialization
        let weights_input_to_layer1 = Array::random(
            (input_size, layer1_size), 
            Uniform::new(-1.0 / (input_size as f32).sqrt(), 1.0 / (input_size as f32).sqrt())
        );
        let weights_layer1_to_layer2 = Array::random(
            (layer1_size, layer2_size), 
            Uniform::new(-1.0 / (layer1_size as f32).sqrt(), 1.0 / (layer1_size as f32).sqrt())
        );
        let weights_layer2_to_output = Array::random(
            (layer2_size, output_size), 
            Uniform::new(-1.0 / (layer2_size as f32).sqrt(), 1.0 / (layer2_size as f32).sqrt())
        );
        
        // Initialize momentum matrices to zeros
        let momentum_input_to_layer1 = Array::zeros((input_size, layer1_size));
        let momentum_layer1_to_layer2 = Array::zeros((layer1_size, layer2_size));
        let momentum_layer2_to_output = Array::zeros((layer2_size, output_size));

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
            momentum_input_to_layer1,
            momentum_layer1_to_layer2,
            momentum_layer2_to_output,
            momentum_factor: 0.9, // Momentum factor
        }
    }

    // Sigmoid activation function
    fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|val| 1.0 / (1.0 + (-val).exp()))
    }

    // Derivative of sigmoid for backpropagation
    fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|val| val * (1.0 - val))
    }

    // Forward propagation.  Returns all of the intermediate and final outputs.
    fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // First layer
        let z1 = input.dot(&self.weights_input_to_layer1);
        let a1 = Self::sigmoid(&z1);
        
        // Second layer
        let z2 = a1.dot(&self.weights_layer1_to_layer2);
        let a2 = Self::sigmoid(&z2);
        
        // Output layer
        let z3 = a2.dot(&self.weights_layer2_to_output);
        let a3 = Self::sigmoid(&z3);
        
        (a1, a2, a3)
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
        // Calculate error at the output
        let output_error = target - final_output;
        
        // Calculate gradient for output layer
        let output_delta = &output_error * &Self::sigmoid_derivative(final_output);
        
        // Update weights for the output layer with momentum
        let layer2_to_output_adjustment = layer2_output.t().dot(&output_delta) * self.learning_rate;
        self.momentum_layer2_to_output = &self.momentum_layer2_to_output * self.momentum_factor + &layer2_to_output_adjustment;
        self.weights_layer2_to_output = &self.weights_layer2_to_output + &self.momentum_layer2_to_output;
        
        // Calculate error and gradient for layer 2
        let layer2_error = output_delta.dot(&self.weights_layer2_to_output.t());
        let layer2_delta = &layer2_error * &Self::sigmoid_derivative(layer2_output);
        
        // Update weights for layer 1 to layer 2 with momentum
        let layer1_to_layer2_adjustment = layer1_output.t().dot(&layer2_delta) * self.learning_rate;
        self.momentum_layer1_to_layer2 = &self.momentum_layer1_to_layer2 * self.momentum_factor + &layer1_to_layer2_adjustment;
        self.weights_layer1_to_layer2 = &self.weights_layer1_to_layer2 + &self.momentum_layer1_to_layer2;
        
        // Calculate error and gradient for layer 1
        let layer1_error = layer2_delta.dot(&self.weights_layer1_to_layer2.t());
        let layer1_delta = &layer1_error * &Self::sigmoid_derivative(layer1_output);
        
        // Update weights for input to layer 1 with momentum
        let input_to_layer1_adjustment = input.t().dot(&layer1_delta) * self.learning_rate;
        self.momentum_input_to_layer1 = &self.momentum_input_to_layer1 * self.momentum_factor + &input_to_layer1_adjustment;
        
        // Apply weight update with tiny L2 regularization to prevent overfitting
        let regularization_factor = 1.0 - 0.00001 * self.learning_rate;
        self.weights_input_to_layer1 = &self.weights_input_to_layer1 * regularization_factor + &self.momentum_input_to_layer1;
    }
    
    // Train the network with a single example
    fn train(&mut self, input: &Array2<f32>, target: &Array2<f32>) {
        let (layer1_output, layer2_output, final_output) = self.forward(input);
        self.backward(input, &layer1_output, &layer2_output, &final_output, target);
    }
    
    // Predict the digit for a given input
    fn predict(&self, input: &Array2<f32>) -> usize {
        let (_, _, output) = self.forward(input);
        
        // Find the index of the maximum value in the output
        let mut max_index = 0;
        let mut max_value = output[[0, 0]];
        
        for i in 1..self.output_size {
            if output[[0, i]] > max_value {
                max_value = output[[0, i]];
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
    
    // Read each row
    for result in reader.records() {
        let record = result?;
        
        // First value is the label
        let label: u8 = record[0].parse()?;
        
        // Rest of the values are pixels
        let mut normalized_pixels = Vec::with_capacity(784);
        for i in 1..record.len() {
            let pixel: u8 = record[i].parse()?;
            normalized_pixels.push(pixel as f32 / 255.0);
        }
        
        // Create a 1x784 array for the input
        let input = Array::from_shape_vec((1, 784), normalized_pixels)?;
        inputs.push(input);
        
        // Create a one-hot encoded target (1x10 array)
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
    
    // Create a neural network with optimized parameters:
    // - 784 input nodes (28x28 pixels)
    // - 400 nodes in first hidden layer (increased for more capacity)
    // - 120 nodes in second hidden layer (increased for more capacity)
    // - 10 output nodes (one for each digit)
    // - Learning rate 0.08 (higher initial rate with momentum)
    let mut nn = NeuralNetwork::new(784, 400, 120, 10, 0.08);
    
    // Train the network for 3 epochs (each example presented 3 times)
    println!("Training the neural network...");
    let epochs = 3;
    
    // Track accuracies during training to see improvement
    let mut epoch_accuracies = Vec::new();
    
    for epoch in 0..epochs {
        println!("Epoch {} of {}", epoch + 1, epochs);
        
        // Shuffle the training data for each epoch to improve learning
        let mut indices: Vec<usize> = (0..train_inputs.len()).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        
        // Adaptive learning rate - adjust for each epoch
        if epoch == 1 {
            nn.learning_rate = 0.04; // Slightly decrease learning rate
            nn.momentum_factor = 0.95; // Increase momentum
        } else if epoch == 2 {
            nn.learning_rate = 0.02; // Further decrease for fine-tuning
            nn.momentum_factor = 0.98; // Further increase momentum
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            nn.train(&train_inputs[idx], &train_targets[idx]);
            
            if (i + 1) % 1000 == 0 {
                println!("  Processed {} of {} training examples", i + 1, train_inputs.len());
            }
        }
        
        // Calculate accuracy after each epoch
        let mut correct = 0;
        for (idx, _) in indices.iter().enumerate().take(1000) { // Use 1000 samples for quick evaluation
            let predicted = nn.predict(&train_inputs[idx]);
            let actual = train_targets[idx].iter().position(|&x| x == 1.0).unwrap_or(0);
            if predicted == actual {
                correct += 1;
            }
        }
        let train_accuracy = (correct as f32) / 1000.0 * 100.0;
        println!("  Training accuracy after epoch {}: {:.2}%", epoch + 1, train_accuracy);
        println!("  Learning rate: {}, Momentum: {}", nn.learning_rate, nn.momentum_factor);
        epoch_accuracies.push(train_accuracy);
    }
    
    // Test the network
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
            // Show intermediate accuracy
            let current_accuracy = (correct_predictions as f32) / (i as f32 + 1.0) * 100.0;
            println!("  Current accuracy: {:.2}%", current_accuracy);
        }
    }
    
    let accuracy = (correct_predictions as f32) / (test_inputs.len() as f32) * 100.0;
    println!("Final Accuracy: {:.2}%", accuracy);
    
    Ok(())
}
