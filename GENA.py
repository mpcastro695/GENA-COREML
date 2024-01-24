import torch
from transformers import AutoTokenizer
import coremltools as ct
import numpy as np

from src.modeling_bert import BertForSequenceClassification
from BertANE import BertForSequenceClassificationANE

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 1. Build the BERT model and load pre-trained weights and tokenizer from Huggingface's Model Hub
tokenizer = AutoTokenizer.from_pretrained('AIRI-Institute/gena-lm-bert-base')
torch_Model = BertForSequenceClassification.from_pretrained('AIRI-Institute/gena-lm-bert-base', return_dict=False).eval()
print(torch_Model.modules)

# 2. Instantiate an ANE optimized model with the base model's configuration
optimized_model = BertForSequenceClassificationANE(torch_Model.config).eval()
print(optimized_model.modules)
optimized_model.load_state_dict(torch_Model.state_dict(), strict=True)


# 3. Remove the final dropout and pooler layers to reveal the last hidden layer sized at [max_Tokens, features(i.e. 768)]
torch_Model = torch.nn.Sequential(*(list(torch_Model.children())[:-2])).eval()
optimized_model = torch.nn.Sequential(*(list(optimized_model.children())[:-2])).eval()

# # (Optional) Save the PyTorch models to disk (compare them with Netron)
# torch.save(torch_Model, 'base_model.pth')
# torch.save(optimized_model, 'optimized_model.pth')

# (Optional) Append a custom layer that returns only the hidden states
class ActivationsOnly(torch.nn.Module):
    def forward(self, input_tuple):
        output = input_tuple[0]
        return output

# (Optional) Append a custom layer that permutes the ouput
class BS1CtoBSC(torch.nn.Module):
    def forward(self, input):
        output = input.permute(0, 3, 1, 2)  # Swaps dimensions to [B, S, 1, C] format
        output = torch.squeeze(output)  # [B, S, C]
        return output

optimized_model = torch.nn.Sequential(*(list(optimized_model.children())), ActivationsOnly(), BS1CtoBSC()).eval()

# 4. Trace the model with dummy inputs to get a TorchScript representation
trace_input = {'input': torch.randint(1, 30000, (1, 512), dtype=torch.int64)}
optimized_trace = torch.jit.trace(optimized_model, example_kwarg_inputs=trace_input)

# 5. Convert the traced model to a Core ML model package and save it to disk
model = ct.convert(
   optimized_trace,
   source='pytorch',
   convert_to='mlprogram',
   inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512, default=256))), dtype=np.int32)],
   outputs=[ct.TensorType(name='features', dtype=np.float32)],
   compute_precision=ct.precision.FLOAT32
)
model.save('GENA.mlpackage')

# # (Optional) Convert and save the NON-optimized model to disk
# unopt_model = torch.nn.Sequential(*(list(torch_Model.children())), ActivationsOnly()).eval()
# unopt_trace = torch.jit.trace(unopt_model, example_kwarg_inputs=trace_input)
# model = ct.convert(
#    unopt_trace,
#    source='pytorch',
#    convert_to='mlprogram',
#    inputs=[ct.TensorType(name='input_ids', shape=ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=512, default=256))), dtype=np.int32)],
#    outputs=[ct.TensorType(name='features', dtype=np.float32)],
#    compute_precision=ct.precision.FLOAT32
# )
# model.save('GENA_UNOPT.mlpackage')

# 6. Load the converted model package from disk and verify its predictions
# Mouse collagen col8a1 exon 1, 413 bp (source: GenBank)
test1_input = 'AGGTGATGGCTGTGCCACCAGGCCCTCTACAGCTGCTGGGAATACTGTTCATCATTTCCCTGAACTCTGTCAGACTCATTCAGGCCGGTGCCTACTATGGAATCAAGCCTCTGCCACCTCAAATCCCTCCTCAGATACCACCACAAATTCCACAGTACCAGCCCTTGGGCCAGCAAGTCCCTCACATGCCTTTGGGCAAAGATGGCCTTTCCATGGGCAAGGAGATGCCTCACATGCAGTATGGCAAAGAGTACCCATACCTCCCCCAATATATGAAGGAAATCCCACCTGTGCCAAGAATGGGCAAGGAAGTGGTGCCCAAAAAAGGCAAAGGTAACATCAATTGAACAGTTTCAAAATAGCTGCTCTCCAGACTTCTAAACTGTAGAAGTTGAGGAGAAAACTATGTAGAC'
# Guinea pig insulin gene, 1472 bp (source: GenBank)
test2_input = 'CTGCAGACCCAGCACCAGGGAAATGATCCAGAAATTGCAACCTCAGCCCCCTGGCCATCTGCTGATGCCACCACCCCCAGGTCCCTAATGGGCCTGGTGGCAGAGTTTGGGAAGATGGGCTCAGGGCTATATAAAGTCCACAAGGACCTAAGAGCCCCCAGTGCTGCTGGGCCAGCTGTATTCTGAGGTGGTCAGCACACAGGTCTGTGTCCTCCGTGCTAGATTGGGGCTGAGAGGCTGGGGGCTCTGGGTTGGCTGGGACAGGACATGGGATTCTTCCTTGTATTGGGGGTTTTGGCTGTTACTCTGTCTCTCCATCAGGTCATCATCCTTTCATCATGGCTCTGTGGATGCATCTCCTCACCGTGCTGGCCCTGCTGGCCCTCTGGGGGCCCAACACTAATCAGGCCTTTGTCAGCCGGCATCTGTGCGGCTCCAACTTAGTGGAGACATTGTATTCAGTGTGTCAGGATGATGGCTTCTTCTATATACCCAAGGACCGTCGGGAGCTAGAGGACCCACAGGGTGAGCCCCTACCTGCCATCCCTGCTGTTTCCGTGCCAGTACCCCAGCTGGCAGGGCATAAGTAAGCAGGAAGCTAATTCCAAGGAGAGTCGATGGGTTTGTTGAAAAGGGAGGCGGCTCTCTTGGTCATTTCGTAAAGTGGTGGTGGCTTCCTATAGCTGCTTTTAAGGGTAAAGGGTAACAGCTGCACCCTTCAGCTGTGGCTTCTGAGCACAACTGGACTCTTCCCTCCACTTGCCTTCGAATGACTGCCCTGGCCTCATGGCAACAGTAGCTCCCTGGTACCAATTTTATTATGCAGATTGCATCTTGGTGTTGATAGCCTTAGGGTAGCCTGGGGGCCATTCATGGGGCGCCCCATCCCTCCTTCCTCCCTGCCTCTGGACAAATGCTCCATGGAGCTCCAAGCTCTGCCACGTGGGAGGTGTGGGTCTCCAGCGCTCTGTGTGCCCAGCATGGCAGCCTCTGTCACCTGGACCAGCTCCCTGGGAGATGCAGTGAGAGGGTGGTAGTGTGGGGCCAGTGCGCAGGCATTCTGCTGCTCCTGACAGCATCTGCCCCTGTCTCTCTCCCCACTGCTGCTGCTCCTGTATTCTGGCACCTCACCCTGCAGTGGAGCAGACAGAACTGGGCATGGGCCTGGGGGCAGGTGGACTACAGCCCTTGGCACTGGAGATGGCACTACAGAAGCGTGGCATTGTGGATCAGTGCTGTACTGGCACCTGCACACGCCACCAGCTGCAGAGCTACTGCAACTAGACACCTGCCTTGAACCTGGCCTCCCACTCTCCCCTGGCAACCAATAAACCCCTTGAATGAGCCCCATTGAATGGTCTGTGTGTCATGGAGGGGGAGGGGCTGACTCAAGGGGGCACATGCATGCCAGCCTATCATCCAGGTTCATTGCAAGACCCCCTCTCTATGCTCTGTGCACCTCTAACACACCC'
# Venus Flytrap Ribonuclease gene, 2434 bp (source: GenBank)
test3_input = 'CGCTCTCTCATCCACGTAAAAGTTGTCTCGCACTTATTTATTCCGTAAACATTGCCGCTCGGTTCATTTACTCATCTGAACTACATCATTCCGTTTATCAATAGCCTTGTCTTAATGTCTCATGACTCCGATCAACACATATATATGCGGACAATGAGGTTAATTTATTCATTTCGGAAATCGGTTGACCTGCTGCCGGAGTATGGTCAAAAGCCCCCATAAGGTAAGACTGCCGACTTGGCATGCATGCCGCATGCGAATTTAAGTAATAAAAGATCAACGATGCGGGTGAGTAATAAACTAATAATTATGATACCTCAGTAGAGTTCGACATTTGATATAATTTAACCAAAATTAAGGATTTTATCTTTATCTGCAACGTTTAATTAATATTTTTCAGGTTGAGATAGATAAAATGATCCGTTAAGATATAGTTGACGTTGAACCAATATCGACAAATTAATTAATGGTTTGGAGCAACCTAATTAACGTAATTAAATTGATCTATATATGCCTAACCATATATATAGGTAACTTAGCTTATAGCTAGTATACTTGAAACGCTTGATTACGATAATTAATAAAATGATGTGAATCATGTGATCGTGTTAAATATGGAAAAGATCACCTTTAATTCCCATTATATAGCCCCTGTATCTGTACGTATAATATCAAGCTATTCTTGAATGGGATTAACTAGCGAAATTAAGTCTTGTAAATTAATAACACGAAACTCTCGATTAATCAATTAGCATATTCCCCCCCTTGCTTGTTAATTAACAGTATATATATGGCTCCCCCTGCAAAGAGCAATCTCATCTCGCTGCTAGGTGGAGATAATCATATATAGATAGATATATATATATATATATATATATTATACCATGAAGAACTCTGTGTTCATCAAGCTTTTGGTTTGGCAATCTCTAGCTGCAGTGGCTCTATGCCAAGGTTTCGACTTCTTCTACTTCGTCCAACAGGTGACTCACGTGCACCATTTACATGATACACCACCATGTTAACTCACGCCATCATGTGTTCAGTAGCTCGTCGTTCGAACGGTTCAACCAAATTGATCGAACAGTTCGGTGGTCTTGATCGAACTGTTAGGTCAACTTAGCCGAACTGCTCGATTCGGCCGAAGTGTTTATAGTCAAGTAGTGAGTTTTTTCTTTGCATTGACTTTGTCTCTCATTTTGAATGTATATATATACATGCAGTGGCCAGGATCATACTGTGACACATCGCAGAGTTGCTGCTATCCTACAACTGGGAAGCCGGCTTCGGATTTTGGCATCCATGGGCTCTGGCCTAACTACAACAGTGGTAGCTATCCATCGAATTGCGACTCCAGCAACCCTTTCGACCCGTCTCAGGTATCTATCTGCATATATACAAAAATGTATGTATGTAAGTACATCGTCGTGATCATATGACATATATAATCACTCTTATGCAGATCCAAGATCTGCTGAGCCAAATGCAAACCGAGTGGCCGTCATTGTCCTGCCCAAGCAGCGATGGTACAAGCTTCTGGACACACGAATGGAACAAGCACGGGACCTGCTCTGAGTCTGTGCTCAATGAACACGCTTACTTTCAAGCCGCTTTGAGCCTCAAGAACTCGTCCAACCTCCTCCAAACCCTAGCTAATGCAGGTAATTAATTAACTAGTACATATAGCAAACTAACTGCGGATGATCATCATTTCAAAGTTTTTAACACAATTATCGGCAGTATAAATATCAATGTGTACCGTGAACAGGAATTACTCCGAATGGCAACTCATATAATTTATCGGACGTGTTGGCCGCGATGAAACAAGCAACTGGAGGATATGATGCTTACATCCAATGCAACACCGACCAGAATGGAAATAGCCAGCTCTACCAAGTATACATGTGTGTTAACACTTCCGGGCAAAGTTTCATCGAATGCCCAGTGGCTCCTAGTCAGAACTGTAACCCCAGCATTGAGTTCCCTTCCTTCTAACCCACCGGCCTCATGCTCATGAGATCTAGCCAGTGAAATGTTAGTTTCATTGTCAATCAGTCCAACAACAACAACAATAAACTAACTCAACAATAAACATATGTTCTTGTATTGATATCAATAGATAGTATAGATAGATGGTGATTGACACATGTATTTGGAATATGATGAATTGTTTTTCATCCAATTTTTACCTAGATTTTACAAACAAATATAGGAAAAATCTTTTTTTTTTTTTAAATTTCATACAAGTGGAGTAACAAATCTAATATGGAACTCACCTTCAAATTGTTCCCCATAATATGAAACTCACCATCCTGTTTTGAAGTTGGGGTAGAGCACGGACTGTTGATTCTCCCCGAGCCAAAATTTCTGCAACAAGAAGAAGAAATACCCACAAAATCAGAACTCTTTATCTTATTCTCATTGGCTGTCCACAAAC'

test1_input = tokenizer(test1_input, return_tensors='pt')
test2_input= tokenizer(test2_input, return_tensors='pt')
test3_input = tokenizer(test3_input, return_tensors='pt')

# Inference with the PyTorch model
with torch.no_grad():
    torchOut1 = torch_Model(test1_input['input_ids'])
    torchOut2 = torch_Model(test2_input['input_ids'])
    torchOut3 = torch_Model(test3_input['input_ids'])

# Inference with Core ML model
model = ct.models.model.MLModel('GENA.mlpackage')
prediction1 = model.predict({'input_ids': test1_input['input_ids'].type('torch.FloatTensor')})
prediction2 = model.predict({'input_ids': test2_input['input_ids'].type('torch.FloatTensor')})
prediction3 = model.predict({'input_ids': test3_input['input_ids'].type('torch.FloatTensor')})

# Extract Core ML tesnors
coreMLTensor1 = prediction1.get('features')
coreMLTensor2 = prediction2.get('features')
coreMLTensor3 = prediction3.get('features')

# Extract torch tensors
torchTensor1 = torchOut1.detach().cpu().numpy() if torchOut1[0].requires_grad else torchOut1[0].cpu().numpy()
torchTensor1 = np.expand_dims(torchTensor1, axis=0)
torchTensor2 = torchOut2.detach().cpu().numpy() if torchOut2[0].requires_grad else torchOut2[0].cpu().numpy()
torchTensor2 = np.expand_dims(torchTensor2, axis=0)
torchTensor3 = torchOut3.detach().cpu().numpy() if torchOut3[0].requires_grad else torchOut3[0].cpu().numpy()
torchTensor3 = np.expand_dims(torchTensor3, axis=0)

# Compare
relTolerance = 1e-02
absTolerance = 4e-02

if np.allclose(coreMLTensor1, torchTensor1, relTolerance, absTolerance):
   print("Test 1 Passed!")
if np.allclose(coreMLTensor2, torchTensor2, relTolerance, absTolerance):
    print("Test 2 Passed!")
if np.allclose(coreMLTensor3, torchTensor3, relTolerance, absTolerance):
    print("Test 3 Passed!")

print("Congrats on the new model!")