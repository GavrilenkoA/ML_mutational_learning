### Antibody escape prediction of mutated SARS-CoV-2 variants.

**Problem statement:**

Since the origin of SARS-Cov2, there has been enough data on RBD variants and their effect on the immune system - whether antibodies bind this variant of RBD or not (binary classification problem).

Machine learning models may be used for predictive profiling on current and prospective variants RBD guide the development of therapeutic antibody treatments and vaccines for COVID-19.

Detailed molecular analysis has revealed that many neutralizing antibodies to SARS-CoV-2 share sequence and structural features. In can be used as a additional data source for pretraining machine learning models for prediction binding RBD with limited labeled data of the antibody of interest. 

In this part of project I implemented fine-tuning approach adding a feature to the RBD sequence with which antibody it interacts. Fine-tuning model used for prediction of binding by therapeutic antibody REGN33.

**Install** **Requirements:**

Create and activate a virtual environment

```bash
pip install -r requirements.txt
```

**Load additional data** 

Load datasets of RBD sequences with physical features

In a root directory:

```bash
mv dataset
```

click to link and download two files:

phys_train: [https://disk.yandex.ru/d/PlJgrISLlixYzA](https://disk.yandex.ru/d/PlJgrISLlixYzA)

phys_test: [https://disk.yandex.ru/d/hXVJLFn4iyWztw](https://disk.yandex.ru/d/hXVJLFn4iyWztw)

**Data description:**

‘Label’ column: 1 - RBD is interacted with antibody, 0 - not.

‘Antibody’ column: type of therapeutic antibody or ACE2 protein.

Datasets:

../dataset/phys_train.csv

../dataset/phys_test.csv   

‘repr’ columns: Every aminoacid in the sequence one-hot encoded and has twenty biochemical and physical features extracted from aaindex database  [https://www.genome.jp/aaindex/](https://www.genome.jp/aaindex/)

../dataset/whole_test.csv   

../dataset/whole_train.csv

‘junction_aa’ column: RBD mutation sequences interacted with four therapeutic antibodies and ACE2 protein

**Utils module**

process_data.py:

*get_desc* - function for converting string to float of one-hot and phisycal features

*get_data*  - Limited data sample of the antibody of interest and full data background antibodies

training_nn.py:

*training_rnn* - trained RNN with embedded antibody data

*training* - Trained models are used with pre-embedded data

evaluation.py:

*plot_loss* - print plot validation and training loss

*measure_metrics* - calculate all classification metrics

*evaluate_model*  - calculates model metrics on embedded antibody test data

*evaluate_model_rnn*  - calculates model metrics on pre-embedded antibody test data

models.py:

*CNN* - 1d convolution neural network process sequential data 

*RNN* - recurrent neural network process already embedded sequence

*RNNembed* - recurrent neural that generate embedding of type antibody or ACE2 protein and process RBD sequence

dataset.py Pytorch datasets:

*Onehot* - one-hot encoded RBD sequence dataset

*OnehotandAB* -  dataset contains a one-hot encoded RBD sequence and the flagged type of antibody

*Phys* - dataset represents the physical and one-hot features of the RBD sequence.

*Abencode1* - dataset generates a special vector for antibody types.

*Abencode2* - dataset generates a special number for each antibody type.

**Notebooks:**

*clas_ML.ipynb* - EDA, train boosting classifiers for antibody escape prediction

*DL_full_df.ipynb* - train neural networks with full data of RBD variants.

*limit_data.ipynb* - train neural networks with limited data of REGN33 data

*CNN_tr_learn.ipynb* - fine-tuning of CNN model using limited data of REGN 33 

*RNN_tr_learn.ipynb* - fine-tuning of RNN model using limited data of REGN 33 

**Results:**

Training RNN model consists of two stages:
1. pre-training using background antibododies
2. fine tuning for prediction of REGN33 antibody escape.

RNN model performance on combined data.
![image](![image](https://github.com/GavrilenkoA/ML_mutational_learning/assets/92908421/b6305f30-92f9-4954-96f2-ab9a5f6fef2c)


Then I provide 40 experiments, sampling different random 100 RBD sequences neutralized by REGN33 antibodies for fine-tuning and compare performance of base model and fine-tuning model, using paired T-test.

![image](https://github.com/GavrilenkoA/ML_mutational_learning/assets/92908421/8344ce73-e615-4f85-91b9-c0924f836953)

#### Conclusion: 
Therapeutic antibodies share similar features - they target the conserved epitopes on RBD and mutations in the same manner reflected by binding RBD and can be used as a data source for pre-training ML models for prediction antibody escape with limited labeled data of antibody of interest.



