use crate::layer::Layerable;

/// The l1.neurons.len() must be equal to l2.weights_len
#[derive(Debug)]
pub struct NeuralNetwork<'a> {
    layers: Vec<&'a dyn Layerable>,
}

impl <'a> NeuralNetwork <'a> {
    pub fn new(layers: Vec<&'a dyn Layerable>) -> Self {
        Self { layers }
    }

    /// Applies an input to a neural-network and returns the output of the last layer.
    pub fn apply(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut last_output = inputs;
        for layer in &self.layers {
            last_output = layer.apply(&last_output);
            println!("Last output: {:?}", last_output);
        }
        last_output
    }
}
