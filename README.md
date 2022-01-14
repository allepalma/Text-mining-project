# Visualizing BERT Embeddings on Adverse Drug Effect Detection
### Laurens Engwegen, Alessandro Palma

## Introduction
This is the repository of the NLP project "Visualizing BERT Embeddings on Adverse Drug Effect Detection". The scope of this work is to compare the performance of three variants of the BERT model (BERT-base, BioBERT and BioClinicalBERT) on the task of Named Entity Recognition of Adverse Drug Effects (ADE) on blog posts from the AskAPatient forum (https://www.askapatient.com/). After comparing the three models in terms of test set performance using different classification heads, we visualize the word embeddings of the last layer of the BERT models to define to what extent they are capable to separate different entities in the latent space. 

<p align="center">
  <img width="400" height="400" src="https://github.com/allepalma/Text-mining-project/blob/main/image/first_page_image.png">
</p>

## Setup
### The data
The `cadec` folder contains the dataset we employed for named entity recognition [1]. The data directory is further divided into sub-folders containing the same number of .txt files. The `meddra`, `sct` and `orginal` folders contain different levels of annotation of entities in the data, with `original` being the one we actively use in the project. More specifically, it classifies relevant tokens as Drug, ADE, Disease, Symptom and Finding. The `text` sub-folder contains the raw posts annotated in the other files. 

### Data processing
Two files are dedicated to data processing for the Named Entity Recognition task:
* `dataset_creation.py`: it contains our custom tokenization process of the text data. The text was split into token words and the labels converted to the BIO-encoding. Then                                results are then stored in the `dataset.txt` file, where sentences are a series of words stacked on top of each other, each associated with an entity-                              label or an O-label.
                         
* `bert_data_creation.py`: it parses `dataset.txt` and prepares the data for its submission to the PyTorch implementation of the BERT models, including the partition into                                    training, validation and test set. 

### The models
The `model` folder contains script for the setup and implementation of the neural models used for named entity recognition.

The `model.py` script contains the implementation of BERT models with different classification heads, namely a linear module, a CRF layer and an LSTM layer. The models are built on the Python3 `transformers`, `pytorch` and `pytorch-crf` libraries. 

The `config.py` file contains a configuration class to set the parameters for the initialization of the BERT models.

### Other scripts
* `main.py`: launches the training loop for the experiments with different BERT models with pre-defined parameters
* `traintest.py`: sets up a class with methods necessary to implement the training process
* `embedding_extractor.py`: reads pre-trained BERT weights and uses them to parameterize an initialized BERT model. Then, such BERT model is employed to extract embeddings from the training instances. The embeddings of multiple word pieces of the same token are averaged to a single vector.

### Other folders
* `logging`: contains the log files with training performances across epochs for all the trained BERT variants.
* `embeddings`: contains pickle files storing the t-SNE embeddings used to produce the plots stored in ` plots` 

## Reproduction
To reproduce the training process of the BERT model, the extraction of the embeddings and the t-SNE dimensional reduction, it is enough to run the following:

```
pip install -r requirements.txt
python main.py 
```
In `main.py` the best BERT model is automatically set to BioclinicalBERT with CRF head, but this can be manually tweaked.


### Packages
The code is fully implemented in Python3 and the required packages to run the provided code can be extrapolated from in `requirements.txt`.

## References
[1] Karimi S, Metke-Jimenez A, Kemp M, Wang C. Cadec: A corpus of adverse drug event annotations. J Biomed Inform. 2015 Jun;55:73-81. doi: 10.1016/j.jbi.2015.03.010. Epub 2015 Mar 27. PMID: 25817970

