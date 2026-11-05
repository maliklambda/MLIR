pub use {
    activation::{ActivationFn, ActivationFnKind, Activationalbe},
    input::WeightedInput,
    transfer::{TransferFn, TransferFnKind, Transferable},
};

mod activation;
mod input;
mod transfer;

#[derive(Debug, Clone)]
pub struct Neuron {
    /// Channels the inputs into a single value.
    /// Produces the transfer-value.
    /// Commonly used: Weighted Sum
    pub transfer_fn: TransferFn,

    /// Transfer-value is checked based on activation.
    /// Bias shifts result independently of the transfer-value.
    /// Commonly used: Stepfn, Sigmoid, etc.
    pub activation_fn: ActivationFn,

    /// The neuron's bias.
    /// Used in activation_fn.
    pub bias: f32,
}

impl Neuron {
    pub fn new(
        transfer_fn: &dyn Transferable,
        activation_fn: &dyn Activationalbe,
        bias: f32,
    ) -> Self {
        Self {
            transfer_fn: transfer_fn.to_transfer_fn(),
            activation_fn: activation_fn.to_activation_fn(),
            bias,
        }
    }

    /// Applies an input to the neuron and returns the resulting output.
    pub fn apply(&self, inputs: &[WeightedInput]) -> f32 {
        let transfer_value = (self.transfer_fn)(inputs);
        (self.activation_fn)(transfer_value, self.bias)
    }
}
