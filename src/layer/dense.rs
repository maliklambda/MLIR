use crate::{layer::Layerable, neuron::Neuron};


/// A single dense layer of a neural network.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// All neurons of the layer.
    pub neurons: Vec<Neuron>,

    /// Store the length of the weight of all neurons' weigths
    pub weights_len: usize,
}

impl DenseLayer {
    /// Builds a layer from a Vec of neurons.
    /// Sets the inputs of every neuron to the shared-inputs.
    pub fn new(neurons: Vec<Neuron>) -> Self {
        assert!(
            !neurons.is_empty(),
            "Cannot build layer from empty neurons-vector."
        );
        let weights_len = neurons.first().unwrap().weights.len();
        // Assert that all neurons' weights are of the same length
        neurons.iter().for_each(|n| {
            assert_eq!(
                n.weights.len(),
                weights_len,
                "Cannot build layer with neurons with weights of different lengths"
            )
        });
        Self {
            neurons,
            weights_len,
        }
    }
}

impl Layerable for DenseLayer {
    /// Applies an input to each neuron to yield a vector of outputs.
    /// For a dense-connected-layer:
    ///     - layer-input-len == number-of-weights-per-neuron
    ///     . layer-output-len == number-of-neurons
    /// The length of the output vector is determined by the number of neurons inside the layer
    ///  -> len(self.neurons) = len(output)
    fn apply(&self, shared_inputs: &[f32]) -> Vec<f32> {
        assert_eq!(
            shared_inputs.len(),
            self.weights_len,
            "Length missmatch for inputs & weights in layer-application."
        );
        self.neurons
            .iter()
            .map(|n| n.apply(shared_inputs))
            .collect()
    }
}
