use crate::neuron::WeightedInput;

/// Channels a vector of weighted inputs into a single transfer-value.
pub type TransferFn = fn(&[WeightedInput]) -> f32;

/// Allows users to specify their own transfer functions.
pub trait Transferable {
    fn to_transfer_fn(&self) -> TransferFn;
}

/// Pre-defined transfer functions.
pub enum TransferFnKind {
    /// Multiplies the weight with the input and returns the sum.
    WeightedSum,

    /// Returns only the maximum value for each input multiplied by its weight.
    Max,

    /// Returns only the minimum value for each input multiplied by its weight.
    Min,
}

/// Transfer functions can be turned into a Transferable
impl Transferable for TransferFnKind {
    /// Returns the concrete function associated with the Transferable
    fn to_transfer_fn(&self) -> TransferFn {
        match self {
            Self::WeightedSum => |inputs| inputs.iter().map(|i| i.input * i.weight).sum(),
            Self::Max => |inputs| {
                inputs
                    .iter()
                    .map(|i| i.input * i.weight)
                    .reduce(f32::max)
                    .unwrap()
            },
            Self::Min => |inputs| {
                inputs
                    .iter()
                    .map(|i| i.input * i.weight)
                    .reduce(f32::min)
                    .unwrap()
            },
        }
    }
}
