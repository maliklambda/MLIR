mod dense;
mod convolutional;
pub use dense::DenseLayer;


/// The trait for layers of a neural network
pub trait Layerable: std::fmt::Debug {
    fn apply(&self, shared_inputs: &[f32]) -> Vec<f32>;
}


#[derive(Debug, Clone)]
pub enum LayerKind {
    /// Dense layers map every output neuron to every input neuron.
    /// -> One-to-one mapping between input- and output-neurons.
    Dense,

    /// Convolutional layers apply filters ("kernels") over the input to extract patterns/features.
    Convolutional,
}
