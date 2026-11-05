pub use {
    activation::{ActivationFn, ActivationFnKind, Activationalbe},
    input::WeightedInput,
    transfer::{TransferFn, TransferFnKind, Transferable},
};

mod activation;
mod input;
mod transfer;

#[derive(Debug)]
pub struct Neuron<'w> {
    /// Inputs to the neuron
    pub inputs: &'w [WeightedInput],

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

impl<'w> Neuron<'w> {
    pub fn new(
        inputs: &'w [WeightedInput],
        transfer_fn: &dyn Transferable,
        activation_fn: &dyn Activationalbe,
        bias: f32,
    ) -> Self {
        Self {
            inputs,
            transfer_fn: transfer_fn.to_transfer_fn(),
            activation_fn: activation_fn.to_activation_fn(),
            bias,
        }
    }

    /// Applies the neuron's input and produces the output.
    pub fn apply(&self) -> f32 {
        let transfer_value = (self.transfer_fn)(self.inputs);
        (self.activation_fn)(transfer_value, self.bias)
    }
}
