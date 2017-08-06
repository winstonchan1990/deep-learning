# Text Generator with GRUs

## About

This directory contains basic template Python scripts to build a text generator using [Gated Recurrent Units (GRUs)](https://arxiv.org/pdf/1412.3555.pdf). 

## Dependencies

* Tensorflow [1.1.0]

## Usage

### 1. Download raw text data

* Download your raw text dataset of choice and place it in the `data/` folder

### 2. Model training

* Run `python scripts/train.py --modelName <name-of-model> --textFilePath </path/to/textdata>`

* Required arguments:
  - `modelName` : Unique name for the model to be trained; all relevant training output will be saved in `/models/<modelName>`
  - `textFilePath` : File path of input text data

* Optional arguments :
  - `seqLen` : Length of character sequence of each training sample
  - `batchSize` : Number of training samples in each batch
  - `internalSize` : Internal size of each GRU cell
  - `numLayers` : Number of GRU layers in the network
  - `learningRate` : Learning rate in training
  - `dropoutKeep` : 1 - dropout rate in training 
  - `numEpochs` : Number of epochs in training
  - `removeNonASCII` : Whether to remove non-ASCII characters from input text data

### 3. Text generation

* Run `python scripts/generate_text.py --modelName <modelName>`.

* When prompted, enter some initial text in the console to initialize the text generator

* Required arguments:
  - `modelName` : Sub-directory in `models/` to retrieve data saved from model training

* Optional arguments:
  - `internalSize` : Internal size of each GRU cell [Must be same as value used in model training]
  - `numLayers` : Number of GRU layers in the network [Must be same as value used in model training]
  - `generatedTextLength` : Number of characters to generate
  - `outfileGeneratedText` : Path to output file to save generated text output


## Example : Generating 'new' Wikipedia articles

* Download the WikiText-2 raw character level dataset from [https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset). Place the `wiki.test.raw` data file in `data/` directory.

* Run `python scripts/train.py --modelName wiki_article_generator --textFilePath data/wiki.test.raw` or `python  scripts/train.py --modelName wiki_article_generator --textFilePath data/wiki.test.raw  --removeNonASCII` (to remove non-ASCII characters)

* Run `python scripts/generate_text.py --modelName wiki_article_generator` to use the trained wiki article text generator

* Here is a sample of what my text generator (trained with 100 epochs) gave me :

```
Enter some text to initiate the text generator: Hello world
...
...
[Text generation start]


Hello worldRavauIarkdooms in Spain and the United Kingdom , they then several turning to Emperor Nero and three
 ships used the eyes of the opening slower than when the Archduke Championship at UFC 82 , forcing i
n sime we 15 of 7 @.@ 5 and $ 7 @.@ 7 metres ( 12 ft 8 in ) wide at 08 : 30  90 ... 713 metres ( 5 @
.@ 0 ft ) he request behind $ 253 @,@ 000 and 5 @,@ 271 season fithing position and the rad coach .
Shout according to the hall bottle , damage was determined , the main sports for additional natural
saw . Aside from a tropical by Cooper Galf of the Ciria Corporation , rabbi initially pierce the Cre
ggan , the battalions departed from the depiction of Pentrage " . Due to its playoff stages was inte
nded to be tood she is usual , whose regain to ensure the scenes . The face is badly experienced in
top team with minor . " American Beauty 's children 's producer is too much light . The face and is
favals assigned to develop the game that Estarles ' Triple Center and the new title was demolished i
n the state charge of the scene on the shortment was formed from a specific task for bitrographerine
s of ancient Poppaea . After published a total of 75 peoples , 33 rung and two death evertly large d
istrict as a reputation for body waters , such as Alexander Express counting on its way , Lesnar , t
he basin on loan lines . Thraw in Taskrony 's planning worker actively received two narrowlers to pr
event the Rockies and Rizal Park .
 The officers along the regard 's AAC seding to see fassing in from the season and the act of a sect
ion .

 = = Egropical cyclone = =

 American Beauty won before 80 @-@ 1800 / and Task Force Manuatta ( 1 @.@ 8 billion ( CNY , $ 279mil
lian frogmen . South African order was somewhat ; either the palate and prepared with the film along
 the rib speed and led to another important positions such as Koufax . Not only do not so feeling of
 penetrations of hill before misroyal buildings in the Marcos Morakia .
 In 2005 , the depression did not want as a tropical storm on November 22 . Nomes ' Charton killed 7
 @.@15 metres ( 5 @.@ 1 ft ) . Anthony concerns , as part of the Fourth Coalition , revealed them es
tablished by the Roys African Commant of Austral in 1869 ; therefore unjaised rapital copyress the p
roposed nature that the Union during their tank unuffine seasons at Surac Islands to a speed of 15 k
norghees , facing a sit @-@ ranked example is thought to imperial services , rather in the 1970s and
 1980s . These include the UN coastal information in superviewed secondare were shacket ; it was dec
ided that the storm mered surges and hodorages , and the Joint Meteorological Agency , Jospeo  aroun
d 1930 .
 Spacey in Amerika after a plastic for the cheek of 779 sestate diment and responded with two ironcl
ads : James Edomine = Eryopoidea = =

...
...
```
... Not quite there yet. Somewhat coherent sentences, but clearly jibberish. We may need to:

* Increase the number of training steps / epochs (perhaps to 1000+ epochs)
* Expand the training corpus


## Acknowledgements

* The code and scripts in this directory are adapted from the following sources:

  - [Martin Gorner](https://github.com/martin-gorner/tensorflow-rnn-shakespeare)
  - [Siraj](https://github.com/llSourcell/wiki_generator_live)

* The WikiText-2 language modelling dataset used in the example was obtained from [Salesforce Research](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset). The following paper introduces the dataset in detail:

> *Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016.*
> [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843)
