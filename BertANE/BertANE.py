import torch

from BertANE.src.modeling_bert import BertEmbeddings, BertLayer, BertEncoder, BertModel, BertForSequenceClassification
from BertANE.LayerNorm import BertLayerNormANE
from BertANE.Dense import BertIntermediateANE, BertOutputANE, BertPoolerANE
from BertANE.Attention import BertAttentionANE

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging

logger = logging.get_logger(__name__)
EPS = 1e-7

class BertEmbeddingsANE(BertEmbeddings):
    """
    BERT Embeddings optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0
    """
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'LayerNorm', BertLayerNormANE(num_channels=config.hidden_size, eps=EPS))

class BertLayerANE(BertLayer):
    """
    BERT Encoder Layer optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0
    """
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
    """
    BERT Encoder Stack w/ layers opimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0
    """
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'layer', torch.nn.ModuleList([BertLayerANE(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_hidden_layers)]))

class BertModelANE(BertModel):
    """
    Full BERT Model optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0
    """
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'embeddings', BertEmbeddingsANE(config))
        setattr(self, 'encoder', BertEncoderANE(config))
        setattr(self, 'pooler', BertPoolerANE(config))


class BertForSequenceClassificationANE(BertForSequenceClassification):
    """
    BERT w/ sequence classification head; optimized for Apple Neural Engine. Compatible w/huggingface transformers 4.17.0
    """
    def __init__(self, config):
        super().__init__(config)
        setattr(self, 'bert', BertModelANE(config))
        setattr(self, 'classifier', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.num_labels, kernel_size=1))
