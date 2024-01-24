# GENA-LM +  Core ML

Scripts for optimizing and converting [GENA-bert-base](https://www.biorxiv.org/content/10.1101/2023.06.12.544594v1) -part of a family of open source models for long DNA sequences- to Apple’s [Core ML](https://developer.apple.com/documentation/coreml) format. GENA-LMs were trained with a masked-language modeling objective across the [T2T](https://www.genome.gov/about-genomics/telomere-to-telomere) human genome assembly. By utilizing Byte-Pair Encoding (BPE) during tokenization, it is capable of handling input sequences up to 4.5 kb long. 


## Environment Setup
You will need an environment with Python 3.8 and the following packages installed:

* PyTorch (used v. 2.0.0)
* Transformers (used v. 4.27.4)
* Core ML Tools (used v. 7.1.0)

If you're on an Apple Silicon Mac, you can clone the Conda environment from the included `environment.yml` file.


## Model Conversion
With dependencies installed, all you have to do is run `GENA-LM.py`. The script will:

1.	Build GENA-LM from `src/modeling_bert.py` and download the pre-trained [weights](https://huggingface.co/AIRI-Institute/gena-lm-bert-base) and tokenizer from Hugging Face’s model hub

2.  Instantiate an optimized version of the model from `BertANE.py` (see *Optimizations*)


3.  Remove the classification heads from both models to reveal their final hidden states', i.e, our 'features'

5.	Trace the optimized model with dummy inputs to get a *TorchScript* representation

6.	Convert the trace with Core ML Tools and save it to disk as a Core ML model package `GENA.mlpackage`

7.	Load the model package from disk and verify its outputs against the original; if the outputs match you should see the message 'Congrats on the new model!'

## Inference
Incorporate the model package, tokenizer and vocab files (`GENA.mlpackage`, `BPTokenizer.swift`, `vocab.json`) into your project. Xcode will automatically generate a Swift class for your model. The tokenizer will load the vocab from file and has functions for tokenizing and byte-pair encoding DNA sequences. The model expects as input up to 510 BP-encoded kmers (plus 2 model tokens), which depending on the sequence, can mean inputs up to 4.5 kb long. 

To extract features from a sequence:

```
let model = try! GENA()
let tokenizer = BPTokenizer.loadTokenizer()!

let exampleInput = “ATGTGTGAATTTAGTAGTCCGCAAATTCCAATAACGGATATTGAGAATGCCATGGAACGGATCGGAAGTCCGGTGAGAGAACTCCGCCGCTTGGATGCGGGGGATGACAGCGAAGTGCTGCTTTGCAATGGGCTGTTTGTCATCAAAATCCCCAAACGGCCATCTGTGCGCGTGACACAGCAAAGAGAATTTGCAGTATACTCCTTTCTCAAACAGTATGATTTACCTGCCTTGATTCCGGAAGTGATTTTTCAATGCAGCGAATTTAATGTTATGTCGTTTATCCCCGGAGAAAACTTTGGCTTTCAAGAATATGCTTTGCTTTCAGAAAAGGAAAAAGAAGCGCTTGCTTCAGATATGGCGATATTTTTGCGGAGATTGCATGGTATATCGGTGCCGCTTTCAGAGAAACCGTTCTGTGAAATCTTCGAAGATAAACGCAAAAGATATTTGGAAGACCAAGAACAGCTGCTTGAAGTGCTCGAAAACCGAAAACTCTTGAATGCACCACTCCAGAAAAATATCCAGACGATATACGAGCATATCGGTCAGAATCAGGAACTGTTTAACTATGCGGCCTGTTTAGTTCACAATGATTTTAGCTCTTCCAATATGGTGTTCAGACATAATCGTCTGTATGGCGTGATCGATTTTGGAGATGTAATTGTCGGCGATCCGGACAATGATTTTTTATGCCTTCTGGATTGCAGCATGGATGACTTTGGGAAAGATTTCGGGCGAAAGGTTTTAAGGCATTATGGCCATCGGAATCCACAATTAGCAGAAAGAAAAGCAGAAATCAATGATGCTTACTGGCCGATACAGCAAGTCCTGCTTGGTGTTCAGAGAGAAGATCGGTCGCTTTTCTGTAAGGGATACCGTGAACTTCTAGCCATAGACCCAGATGCTTTCATTTTATAA”

let encodedInput = tokenizer.tokenize(exampleInput) //159 token IDs

guard let output = try? model.prediction(input_ids: encodedInput) else {
    fatalError("Unexpected runtime error.")
}
print(output.features)
```
The output is a [MLMultiArray](https://developer.apple.com/documentation/coreml/mlmultiarray) of size [1, maxTokens, features], i.e., [1, 512, 768]. Please refer to the `GENA Demo` project for more details on using the converted model.

## Optimizations
The pre-trained model can be converted as-is with Core ML Tools. However, the resulting .mlpackage incurs **large** memory spikes during inference. Following guidance from [this](https://machinelearning.apple.com/research/neural-engine-transformers) paper, the following changes were made prior to conversion:
- Linear (dense) layers were replaced with their 2D convolution equivalent
- Layer normalization was replaced with an [optimized](https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/layer_norm.py) equivalent
- Self-attention modules were replaced with an [optimized](https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/multihead_attention.py) equivalent

Weights are reshaped to match the layer changes by registering pre-hooks, per the paper. With these changes, GENA sees up to **50x** reduction in memory usage. Prediction latency remains largely unchanged. 


## GPU / ANE Support?

The model is currently entirely CPU-bound. Running the model in an Xcode project will produce the following message:

<img src="GENA Demo/GENA Demo/Assets.xcassets/Warning.imageset/Screenshot.png" width="600" />


![Alt text](<GENA Demo/GENA Demo/Assets.xcassets/Features.imageset/Screenshot.png>)