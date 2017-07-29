# Text Generator with GRUs

## About

This directory contains basic template Python scripts to build a text generator using [Gated Recurrent Units (GRUs)](https://arxiv.org/pdf/1412.3555.pdf). 

## Dependencies

* Tensorflow [1.1.0]

## Usage

### 1. Download raw text data

* Download your raw text dataset of choice and place it in the `data/` folder

### 2. Model training

* Run `python scripts/train.py` in this directory
* Configurable options :
  - `seqLen` : Length of character sequence of each training sample
  - `batchSize` : Number of training samples in each batch
  - `internalSize` : Internal size of each GRU cell
  - `numLayers` : Number of GRU layers in the network
  - `learningRate` : Learning rate in training
  - `dropoutKeep` : 1 - dropout rate in training 
  - `numEpochs` : Number of epochs in training
  - `textFilePath` : File path of input text data
  - `removeNonASCII` : Text preprocessing option - whether to remove non-ASCII characters

### 3. Text generation

(In progress...)


## Example : Generating 'new' Wikipedia articles

* Download the WikiText-2 raw character level dataset from [https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset). Place the `wiki.test.raw` data file in `data/` directory.

* Run `python scripts/train.py --textFilePath data/wiki.test.raw` or `python  scripts/train.py --textFilePath data/wiki.test.raw  --removeNonASCII` (to remove non-ASCII characters)

* A sample training log can be viewed in `logfiles/trainlog-wiki-test-raw.txt`. By epoch 3, the text generator should be able to start generating legible words and sentences.


## Acknowledgements

* The code and scripts in this directory are adapted from the following sources:

  - [Martin Gorner](https://github.com/martin-gorner/tensorflow-rnn-shakespeare)
  - [Siraj](https://github.com/llSourcell/wiki_generator_live)

* The WikiText-2 language modelling dataset used in the example was obtained from [Salesforce Research](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset). The following paper introduces the dataset in detail:

> *Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016.*
> [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)
