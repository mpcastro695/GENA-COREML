import torch

def linear_to_conv2d(state_dict, prefix=None, local_metadata=None, strict=True, missing_keys=None, unexpected_keys=None, error_msgs=None):
    """
     Returns a state_dict where the weights of linear layers are unsqueezed twice to fit
     the Conv2D, ANE-optimized drop-ins.
    """
    modified_state_dict = state_dict.copy()

    for k in state_dict:
        is_key = all(substr in k for substr in ['key', '.weight'])
        is_query = all(substr in k for substr in ['query', '.weight'])
        is_value = all(substr in k for substr in ['value', '.weight'])

        is_internal_proj = all(substr in k for substr in ['dense', '.weight'])
        is_output_proj = all(substr in k for substr in ['classifier', '.weight'])

        is_pooler = all(substr in k for substr in ['pooler', '.weight'])
        # is_embedding = all(substr in k for substr in ['Embeddings', '.weight'])

        if is_key or is_query or is_value or is_internal_proj or is_output_proj or is_pooler:
            print(f'Weights for {k} unsqueezed twice to match data format expected in ANE optimized Conv2d layers')
            modified_state_dict[k] = torch.unsqueeze(modified_state_dict[k], dim=2).contiguous()
            modified_state_dict[k] = torch.unsqueeze(modified_state_dict[k], dim=3).contiguous()

    return modified_state_dict

def correct_for_bias_scale_order_inversion(state_dict, prefix=None, local_metadata=None, strict=True, missing_keys=None, unexpected_keys=None, error_msgs=None):
    """
    Note: torch.nn.LayerNorm and ane_transformers.reference.layer_norm.LayerNormANE
    apply scale and bias terms in opposite orders. In order to accurately restore a
    state_dict trained using the former into the the latter, we adjust the bias term
    """

    if state_dict[prefix + 'bias'] != None and state_dict[prefix + 'bias'] != None:
        state_dict[prefix + 'bias'] = state_dict[prefix + 'bias'] / state_dict[prefix + 'weight']
        print(f'Weights for Layer Norm {prefix} Inverted to match data format expected in ANE optimized module')

    return state_dict
