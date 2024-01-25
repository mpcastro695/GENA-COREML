import torch
from src.modeling_bert import BertEmbeddings, BertIntermediate, BertSelfOutput, BertOutput, BertAttention, BertLayer, BertEncoder, BertModel, BertPooler, BertForSequenceClassification

EPS = 1e-7

def linear_to_conv2d(state_dict, prefix=None, local_metadata=None, strict=True, missing_keys=None, unexpected_keys=None, error_msgs=None):
    """
     Returns a BERT state_dict where the weights of linear layers are unsqueezed twice to fit
     their Conv2D, ANE-optimized equivalents.
    """

    for k in state_dict:
        is_key = all(substr in k for substr in ['key', '.weight'])
        is_query = all(substr in k for substr in ['query', '.weight'])
        is_value = all(substr in k for substr in ['value', '.weight'])

        is_internal_proj = all(substr in k for substr in ['dense', '.weight'])
        is_output_proj = all(substr in k for substr in ['classifier', '.weight'])

        is_pooler = all(substr in k for substr in ['pooler', '.weight'])

        if is_key or is_query or is_value or is_internal_proj or is_output_proj or is_pooler:
            print(f'Weights for {k} unsqueezed twice to match data format expected in ANE optimized Conv2d layers')
            state_dict[k] = torch.unsqueeze(state_dict[k], dim=2).contiguous()
            state_dict[k] = torch.unsqueeze(state_dict[k], dim=3).contiguous()

def correct_for_bias_scale_order_inversion(state_dict, prefix=None, local_metadata=None, strict=True, missing_keys=None, unexpected_keys=None, error_msgs=None):
    """
    Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
    apply scale and bias terms in opposite orders. In order to accurately restore a
    state_dict trained using the former into the the latter, we adjust the bias term
    """

    if state_dict[prefix + 'weight'] != None and state_dict[prefix + 'bias'] != None:
        state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
        print(f'Weights for Layer Norm {prefix} Inverted to match data format expected in ANE optimized module')

    return state_dict

class LayerNormANE(torch.nn.Module):
    """
    Layer Normalization optimized for Apple Neural Engine (ANE) execution. Please refer to the Apple Machine Learning
    research paper 'Deploying Transformers on the Apple Neural Engine for the original code.
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

        # assert input_rank == self.expected_rank
        # assert inputs.size(1) == self.num_channels

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Registers the pre_hook to properly restore LayerNorm scale and bias from a pre-trained BERT state dictionary
        self._register_load_state_dict_pre_hook(correct_for_bias_scale_order_inversion)

class BertEmbeddingsANE(BertEmbeddings):
    # Hugging Face 4.17 BERT Embeddings adapter class
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'LayerNorm', BertLayerNormANE(num_channels=config.hidden_size, eps=EPS))

class BertIntermediateANE(BertIntermediate):
    # Hugging Face 4.17 BERT Intermediate adapter class
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.intermediate_size, kernel_size=1))

class BertSelfOutputANE(BertSelfOutput):
    # Hugging Face 4.17 BERT Self Output adapter class
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1))

class BertOutputANE(BertOutput):
    # Hugging Face 4.17 BERT Output adapter class
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.intermediate_size, out_channels=config.hidden_size, kernel_size=1))

class SelfAttentionANE(torch.nn.Module):
    """
    Self Attention optimized for efficient ANE deployment. Please refer to the Apple Machine Learning
    research paper 'Deploying Transformers on the Apple Neural Engine for the original code.'
    """
    def __init__(self, embed_dim, d_qk=None, d_v=None, d_out=None, n_head=8, dropout=0.1, **kwargs):
        """
        Args:
            embed_dim:          Dimensionality of the input embeddings
            d_qk:               Dimensionality of the query and key embeddings. They must match in order to compute
                                dot product attention. If None, it is set to that of the input tensors, i.e. `embed_dim`
            d_v:                Dimensionality of the value embeddings. It may differ from that of the query and
                                key embeddings. If None, it is set to that of the input tensors.
            d_out:              Dimensionality of the output projection. If None, it is set to that of the input tensors
            n_head:             The number of different attention heads to compute in parallel, uses :math:`d_qk/n_head`
                                channel groups in parallel to learn different attention patterns in a single layer
            dropout:            The probability that each attention weight is zero-masked independent of other weights
        """
        super().__init__()

        self.d_qk = d_qk or embed_dim
        self.d_v = d_v or embed_dim
        self.d_out = d_out or embed_dim

        self.n_head = n_head
        if self.d_qk % self.n_head != 0 or self.d_v % self.n_head != 0:
            raise ValueError(
                f"Either query-key dimensions ({self.d_qk}) or the value embeddings "
                f"dimensions ({self.d_v}) is not divisible by n_head ({self.n_head})"
            )
        self.q_normalize_fact = float(self.d_qk // self.n_head) ** -0.5

        self.query = torch.nn.Conv2d(embed_dim, self.d_qk, 1)
        self.value = torch.nn.Conv2d(embed_dim, self.d_v, 1)
        self.key = torch.nn.Conv2d(embed_dim, self.d_qk, 1)

        # self.out_proj = torch.nn.Conv2d(self.d_v, self.d_out, 1) #Replace with ANE_BertSelfOutput
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0. else torch.nn.Identity()

        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.constant_(module.bias, 0.)

    def _attention_fn(self, q, k, v, qk_mask, k_mask, return_weights):
        """Core routine for computing multi-head attention

        Args:
            q:              Projected query embeddings of shape (batch_size, d_qk, 1, tgt_seq_len)
            k:              Projected key embeddings of shape (batch_size, d_qk, 1, src_seq_len)
            v:              Projected value embeddings of shape (batch_size, d_v, 1, src_seq_len)
            qk_mask:        Float tensor of shape (batch_size, src_seq_len, 1, tgt_seq_len).
                            Indices with the a high negative value, e.g. -1e4, are excluded from attention
            k_mask:         Float tensor of shape (batch_size, src_seq_len, 1, 1).
                            Indices with the a high negative value, e.g. -1e4, are excluded from attention

        Returns:
            attn:           Attention embeddings of shape (batch_size, d_v, 1, tgt_seq_len)
            attn_weights:   If `return_weights` is True, returns the softmax attention weights used to compute the attention matrix
        """

        # Principle 2: Chunking Large Intermediate Tensors  (machinelearning.apple.com/research/apple-neural-engine)
        # Split q, k and v to compute a list of single-head attention functions
        mh_q = q.split(
            self.d_qk // self.n_head,
            dim=1)  # n_head * (batch_size, d_qk/n_head, 1, tgt_seq_len)
        # Principle 3: Minimizing Memory Copies
        # Avoid as many transposes and reshapes as possible
        mh_k = k.transpose(1, 3).split(
            self.d_qk // self.n_head,
            dim=3)  # n_head * (batch_size, src_seq_len, 1, d_qk/n_head)
        mh_v = v.split(
            self.d_v // self.n_head,
            dim=1)  # n_head * (batch_size, d_v/n_head, 1, src_seq_len)

        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * self.q_normalize_fact
            for qi, ki in zip(mh_q, mh_k)
        ]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        # Apply attention masking
        if qk_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + qk_mask
        if k_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + k_mask

        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)
        mh_w = [self.dropout(aw) for aw in attn_weights]  # n_head * (batch_size, src_seq_len, 1, tgt_seq_len)

        attn = [torch.einsum('bkhq,bchk->bchq', wi, vi) for wi, vi in zip(mh_w, mh_v)]  # n_head * (batch_size, d_v/n_head, 1, tgt_seq_len)
        attn = torch.cat(attn, dim=1)  # (batch_size, d_v, 1, tgt_seq_len)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def _forward_impl(self, q, k, v, qpos=None, kpos=None, vpos=None, qk_mask=None, k_mask=None, return_weights=True,):
        """
        Args:
            q:                  Query embeddings of shape (batch_size, embed_dim, 1, tgt_seq_len)
            k:                  Key embeddings of shape (batch_size, embed_dim, 1, src_seq_len)
            v:                  Value embeddings of shape (batch_size, embed_dim, 1, src_seq_len)
            qpos:               Positional encodings for the query embeddings with same shape as `q`
            kpos:               Positional encodings for the key embeddings with same shape as `k`
            vpos:               Positional encodings for the key embeddings with same shape as `v`
            qk_mask:            Float tensor with shape (batch_size, src_seq_len, 1, tgt_seq_len). Example use case: for causal masking
                                in generative language models (e.g. GPT), fill the upper triangular part with a high negative value (e.g. -1e4).
                                Indices with the a high negative value, e.g. -1e4, are excluded from attention
            k_mask:             Float tensor with shape (batch_size, src_seq_len, 1, 1). Example use case: when excluding embeddings that
                                correspond to zero-padded pixels in an image or unused tokens in a text token sequence from attention.
                                Indices with the a high negative value, e.g. -1e4, are excluded from attention
            return_weights:     If True, returns the intermediate attention weights

        Note: If any of q,k,v has shape (batch_size, embed_dim, height, width) that represent a 2-d feature map, this will
        be flattened to (batch_size, embed_dim, 1, height * width)

        Note: `attn_weights` are never passed downstream even when return_weights=True because all the attn_weights
        are harvested from the outermost module (e.g. ane_transformers.model#Transformer) by means of forward hooks
        """

        # Parse tensor shapes for source and target sequences
        # assert len(q.size()) == 4 and len(k.size()) == 4 and len(v.size()) == 4
        b, ct, ht, wt = q.size()
        b, cs, hs, ws = k.size()

        tgt_seq_len = ht * wt
        src_seq_len = hs * ws

        # Add positional encodings if any
        if qpos is not None:
            q = q + qpos
        if kpos is not None:
            k = k + kpos
        if vpos is not None:
            v = v + kpos

        # Project q,k,v
        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        # Validate qk_mask (`attn_mask` in `torch.nn.MultiheadAttention`)
        expected_qk_mask_shape = [b, src_seq_len, 1, tgt_seq_len]
        if qk_mask is not None:
            if qk_mask.dtype != torch.float32:
                raise RuntimeError(
                    f"`qk_mask` must be of type torch.float32, received {qk_mask.dtype}"
                )
            if list(qk_mask.size()) != expected_qk_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `qk_mask` (Expected {expected_qk_mask_shape}, got {list(qk_mask.size())}"
                )

        # Validate k_mask (`key_padding_mask` in `torch.nn.MultiheadAttention`)
        expected_k_mask_shape = [b, src_seq_len, 1, 1]
        if k_mask is not None:
            if k_mask.dtype != torch.float32:
                raise RuntimeError(
                    f"`k_mask` must be of type torch.float32, received {k_mask.dtype}"
                )
            if list(k_mask.size()) != expected_k_mask_shape:
                raise RuntimeError(
                    f"Invalid shape for `k_mask` (Expected {expected_k_mask_shape}, got {list(k_mask.size())}"
                )

        # Call the attention function
        attn, attn_weights = self._attention_fn(q, k, v, qk_mask, k_mask, return_weights)

        # Revert to original dimension permutation
        attn = attn.contiguous().view(b, self.d_v, ht, wt)
        # attn = self.out_proj(attn)

        return attn, attn_weights

    def forward(self, q, k, v, **kwargs):
        return self._forward_impl(q, k, v, **kwargs)

class BertAttentionANE(BertAttention):
    # Hugging Face 4.17 BERT Attention adapter class
    def __init__(self, config):
        # Initialize the ANE_Attention super class with the embedding dimensions
        ## Sets 'self' attribute to be an instance of ANE_SelfAttention
        super().__init__(config)
        setattr(self, 'config', config)
        setattr(self, 'self', SelfAttentionANE(config.hidden_size, n_head=config.num_attention_heads))
        setattr(self, 'softmax', torch.nn.Softmax(dim=-1))
        setattr(self, 'output', BertSelfOutputANE(config))

    def forward(self, qkv, **kwargs):
        # Chunks the input and calculates raw attention scores
        attn, attn_weights = self.self._forward_impl(qkv, qkv, qkv, **kwargs)
        # Feeds the attention scores to a dense layer
        attn_output = self.output(attn, qkv)

        return attn_output, attn_weights


class BertLayerANE(BertLayer):
    # Hugging Face 4.17 BERT Layer adapter class
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(config)
        self.config = config
        setattr(self, 'pre_attention_ln', BertLayerNormANE(num_channels=config.hidden_size))
        setattr(self, 'post_attention_ln', BertLayerNormANE(num_channels=config.hidden_size))
        setattr(self, 'attention', BertAttentionANE(config))
        setattr(self, 'intermediate', BertIntermediateANE(config))
        setattr(self, 'output', BertOutputANE(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        position_bias=None,
        output_attentions=False,
    ):
        # Normalizes the input and calculates attention w/ ANE-optimized module
        # Adds it back to the input ('i.e, our 'hidden states')
        attn, attn_weights = self.attention(hidden_states if not self.pre_layer_norm else self.pre_attention_ln(hidden_states))
        attn_output = hidden_states + attn

        # Normalizes the output from the attention sublayer and feeds it to the FFN modules
        intermediate_output = self.intermediate(attn_output if not self.pre_layer_norm else self.post_attention_ln(attn_output))
        dense_output = self.output(intermediate_output, attn)

        return dense_output + attn_output, attn_weights

class BertEncoderANE(BertEncoder):
    # Hugging Face 4.17 BERT Encoder adapter class
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'layer', torch.nn.ModuleList([BertLayerANE(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_hidden_layers)]))

class BertPoolerANE(BertPooler):
    # Hugging Face 4.17 BERT Pooler adapter class
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1))

    def forward(self, hidden_states):
        # "Pool" the model by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[0]
        # pooled_output = self.dense(first_token_tensor)
        # pooled_output = self.activation(pooled_output)
        # return pooled_output

        # Weights for the pooler head are randomly initialized
        return first_token_tensor

class BertModelANE(BertModel):
    # Hugging Face 4.17 BERT Model adapter class
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'embeddings', BertEmbeddingsANE(config))
        setattr(self, 'encoder', BertEncoderANE(config))
        setattr(self, 'pooler', BertPoolerANE(config))

class BertForSequenceClassificationANE(BertForSequenceClassification):
    # Hugging Face 4.17 Bert Classification head adapter class
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'bert', BertModelANE(config))
        setattr(self, 'classifier', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.num_labels, kernel_size=1))

        # Registers the pre-hook that reshapes linear weights to Conv2D weights
        self._register_load_state_dict_pre_hook(linear_to_conv2d)