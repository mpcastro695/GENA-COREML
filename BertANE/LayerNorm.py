import torch

from BertANE.WeightUtils import correct_for_bias_scale_order_inversion

class LayerNormANE(torch.nn.Module):
    """
    Layer Normalization optimized for Apple Neural Engine (ANE) execution
    """
    def __init__(self, num_channels, clip_mag=None, eps=1e-5, elementwise_affine=True):
        """
        Args:
            num_channels:       Number of channels (C) where the expected input data format is BC1S. S stands for sequence length.
            clip_mag:           Optional float value to use for clamping the input range before layer norm is applied.
                                If specified, helps reduce risk of overflow.
            eps:                Small value to avoid dividing by zero
            elementwise_affine: If true, adds learnable channel-wise shift (bias) and scale (weight) parameters
        """
        super().__init__()

        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        self.expected_rank = len('BC1S')

        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = torch.nn.Parameter(torch.Tensor(num_channels))
            self.bias = torch.nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, inputs):
        input_rank = len(inputs.size())

        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # Migrate the data format from BSC to BC1S (most conducive to ANE)
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels

        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom

        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)
                   ) * self.weight.view(1, self.num_channels, 1, 1)

        return out

class BertLayerNormANE(LayerNormANE):
    """
    A wrapper layer that registers a pre-hook to modify the LayerNorm weights of a pre-trained BERT checkpoint.
    Compatible w/ huggingface transformers 4.17.0
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Registers a pre_hook to properly restore LayerNorm scale and bias from pre-trained models
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)