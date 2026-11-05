use crate::neuron::{Neuron, WeightedInput};


/// A single layer of a neural network.
#[derive(Debug, Clone)]
pub struct Layer <'w> {
    /// The inputs are shared between all neurons.
    /// Since the inputs are set for each neuron
    pub shared_inputs: &'w [WeightedInput],

    /// All neurons of the layer.
    pub neurons: Vec<Neuron>,
}

impl <'w> Layer <'w> {
    /// Builds a layer from a Vec of neurons.
    /// Sets the inputs of every neuron to the shared-inputs.
    pub fn new(shared_inputs: &'w[WeightedInput], neurons: Vec<Neuron>) -> Self {
        Self {
            shared_inputs,
            neurons,
        }
    }

    /// Applies the inputs to each neuron to yield a vector of outputs.
    pub fn apply(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.apply(self.shared_inputs)).collect()
    }
}
