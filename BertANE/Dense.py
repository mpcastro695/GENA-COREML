import torch

from BertANE.src.modeling_bert import BertIntermediate, BertOutput, BertSelfOutput, BertPooler
            
class BertIntermediateANE(BertIntermediate):
    """ Dense module optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0"""
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.intermediate_size, kernel_size=1))

class BertSelfOutputANE(BertSelfOutput):
    """ Dense module optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0"""
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=1))

class BertOutputANE(BertOutput):
    """ Dense module optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0"""
    def __init__(self, config):
        super().__init__(config)
        self.seq_len_dim = 3
        setattr(self, 'dense', torch.nn.Conv2d(in_channels=config.intermediate_size, out_channels=config.hidden_size, kernel_size=1))

class BertPoolerANE(BertPooler):
    """ Dense module optimized for Apple Neural Engine. Compatible w/ huggingface transformers 4.17.0"""
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
