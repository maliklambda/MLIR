use crate::layer::Layer;

/// The l1.neurons.len() must be equal to l2.weights_len
#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Applies an input to a neural-network and returns the output of the last layer.
    pub fn apply(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut last_output = inputs;
        for layer in &self.layers {
            last_output = layer.apply(&last_output);
        }
        last_output
    }
}
