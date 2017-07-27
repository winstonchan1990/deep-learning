# Text Generator with GRUs

## About

This directory contains basic template Python scripts to build a text generator using [Gated Recurrent Units (GRUs)](https://arxiv.org/pdf/1412.3555.pdf). 

## Dependencies

* Tensorflow (1.1.0)

## Usage

### 1. Download raw text data

* Download your raw text dataset of choice and place it in the `data/` folder

### 2. Model training

* In `train.py`, change `textfilepath` to the relative file path of your raw text dataset
* Run `python scripts/train.py` in this directory

### 3. Text generation

(In progress...)

## Acknowledgements

* The code and scripts in this directory are adapted from the following sources:

  - [Martin Gorner](https://github.com/martin-gorner/tensorflow-rnn-shakespeare)
  - [Siraj](https://github.com/llSourcell/wiki_generator_live)
