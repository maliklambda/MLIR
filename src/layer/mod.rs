use crate::neuron::{Neuron, WeightedInput};

/// A single layer of a neural network.
#[derive(Debug, Clone)]
pub struct Layer {
    /// All neurons of the layer.
    pub neurons: Vec<Neuron>,
}

impl Layer {
    /// Builds a layer from a Vec of neurons.
    /// Sets the inputs of every neuron to the shared-inputs.
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }

    /// Applies an input to each neuron to yield a vector of outputs.
    pub fn apply(&self, shared_inputs: &[WeightedInput]) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|n| n.apply(shared_inputs))
            .collect()
    }
}
