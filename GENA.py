import torch
from transformers import AutoTokenizer
import coremltools as ct
import numpy as np

from src.modeling_bert import BertForSequenceClassification
from BertANE import BertForSequenceClassificationANE

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Build the BERT model and download the pre-trained weights and tokenizer from Huggingface's Model Hub
tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
torch_model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base', return_dict=False).eval()

# 2. Instantiate an ANE-optimized model
optimized_model = BertForSequenceClassificationANE(torch_model.config).eval()
optimized_model.load_state_dict(torch_model.state_dict(), strict=True)
print(optimized_model.modules)

# 3. Remove the untrained dropout and classifier head
torch_model = torch.nn.Sequential(*(list(torch_model.children())[:-2])).eval()
optimized_model = torch.nn.Sequential(*(list(optimized_model.children())[:-2])).eval()
print(optimized_model.modules)

# (Optional) Append a custom layer that returns only the hidden states
class ActivationsOnly(torch.nn.Module):
    def forward(self, input_tuple):
        output = input_tuple[0]
        return output

# (Optional) Append a custom layer that permutes the ouput back to BSC format
class BS1CtoBSC(torch.nn.Module):
    def forward(self, input):
        output = input.permute(0, 3, 1, 2)  # Swaps dimensions to [B, S, 1, C] format
        output = torch.squeeze(output)  # [B, S, C]
        return output

optimized_model = torch.nn.Sequential(*(list(optimized_model.children())), ActivationsOnly(), BS1CtoBSC()).eval()

# 4. Trace the optimized model with dummy inputs to get a TorchScript representation
trace_input = {'input': torch.randint(1, 30000, (1, 512), dtype=torch.int64)}
optimized_trace = torch.jit.trace(optimized_model, example_kwarg_inputs=trace_input)

# 5. Convert the trace to a FP32 Core ML model package and save it to disk
model_f32 = ct.convert(
   optimized_trace,
   source='pytorch',
   convert_to='mlprogram',
   inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, 512)), dtype=np.int32)],
   outputs=[ct.TensorType(name='features', dtype=np.float32)],
   compute_precision=ct.precision.FLOAT32
)
model_f32.save('GENA_FP32.mlpackage')

# Convert the trace to a FP16 model package and save it to disk
model_f16 = ct.convert(
   optimized_trace,
   source='pytorch',
   convert_to='mlprogram',
   inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, 512)), dtype=np.int32)],
   outputs=[ct.TensorType(name='features', dtype=np.float32)],
   compute_precision=ct.precision.FLOAT16
)
model_f16.save('GENA_FP16.mlpackage')

# 6. Load the converted model package from disk and verify its prediction
# Mouse collagen col8a1 exon 1, 413 bp (source: GenBank)
test_input = 'AGGTGATGGCTGTGCCACCAGGCCCTCTACAGCTGCTGGGAATACTGTTCATCATTTCCCTGAACTCTGTCAGACTCATTCAGGCCGGTGCCTACTATGGAATCAAGCCTCTGCCACCTCAAATCCCTCCTCAGATACCACCACAAATTCCACAGTACCAGCCCTTGGGCCAGCAAGTCCCTCACATGCCTTTGGGCAAAGATGGCCTTTCCATGGGCAAGGAGATGCCTCACATGCAGTATGGCAAAGAGTACCCATACCTCCCCCAATATATGAAGGAAATCCCACCTGTGCCAAGAATGGGCAAGGAAGTGGTGCCCAAAAAAGGCAAAGGTAACATCAATTGAACAGTTTCAAAATAGCTGCTCTCCAGACTTCTAAACTGTAGAAGTTGAGGAGAAAACTATGTAGAC'
test_input = tokenizer(test_input, return_tensors='pt', max_length=512, padding='max_length')

# Inference with the PyTorch model
with torch.no_grad():
    torch_output = torch_model(test_input['input_ids'])

# Inference with Core ML models
model_f32 = ct.models.model.MLModel('GENA_FP32.mlpackage')
prediction_f32 = model_f32.predict({'input_ids': test_input['input_ids'].type('torch.FloatTensor')})
core32_tensor = prediction_f32.get('features')

model_f16 = ct.models.model.MLModel('GENA_FP16.mlpackage')
prediction_f16 = model_f16.predict({'input_ids': test_input['input_ids'].type('torch.FloatTensor')})
core16_tensor = prediction_f16.get('features')

# Extract torch tensors
torch_tensor = torch_output.detach().cpu().numpy() if torch_output[0].requires_grad else torch_output[0].cpu().numpy()
torch_tensor = np.squeeze(torch_tensor)

# Compare the output activations
relTolerance = 1e-02
absTolerance = 1e-03
np.testing.assert_allclose(core32_tensor, torch_tensor, relTolerance, absTolerance)
print("Congrats on the new FP32 model!")

relTolerance = 2e+01
absTolerance = 1e-01
np.testing.assert_allclose(core16_tensor, torch_tensor, relTolerance, absTolerance)
print("Congrats on the new FP16 model!")