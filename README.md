### Antibody escape prediction of mutated SARS-CoV-2 variants.
Authors:

Alexander Gavrilenko
Daria Balashova

**Problem statement:**

Since the origin of SARS-Cov2, there has been enough data on RBD variants and their effect on the immune system - whether antibodies bind this variant of RBD or not (binary classification problem).

Machine learning models may be used for predictive profiling on current and prospective variants RBD guide the development of therapeutic antibody treatments and vaccines for COVID-19.

Detailed molecular analysis has revealed that many neutralizing antibodies to SARS-CoV-2 share sequence and structural features. In can be used as a additional data source for pretraining machine learning models for prediction binding RBD with limited labeled data of the antibody of interest. 

In this part of project I implemented fine-tuning approach adding a feature to the RBD sequence with which antibody it interacts. Fine-tuning model used for prediction of binding by therapeutic antibody REGN33.

**Usage**

Create, activate a virtual environment and install needed libraries.

```bash
python3.10 -m venv <virtual-environment-name>
source env/bin/activate
pip install -r requirements.txt
```

**Additional data** 

Load datasets of RBD sequences with physical features

In a root directory:

```bash
mv dataset
```

click to link and download two files:

phys_train: [https://disk.yandex.ru/d/PlJgrISLlixYzA](https://disk.yandex.ru/d/PlJgrISLlixYzA)

phys_test: [https://disk.yandex.ru/d/hXVJLFn4iyWztw](https://disk.yandex.ru/d/hXVJLFn4iyWztw)

**Data description:**

 ‘Label’ column: labels corresponding to whether the antibody binds this variant of RBD or not

  ‘Antibody’ column: type of therapeutic antibody or ACE2 protein.

  Datasets:

   ../dataset/phys_train.csv

   ../dataset/phys_test.csv   

   ‘repr’ columns: Every aminoacid in the sequence one-hot encoded and has twenty biochemical and physical features extracted from aaindex                 database  [https://www.genome.jp/aaindex/](https://www.genome.jp/aaindex/)

   ../dataset/whole_test.csv   

   ../dataset/whole_train.csv

   ‘junction_aa’ column: RBD mutation sequences interacted with four therapeutic antibodies and ACE2 protein

**Utils module**


[process_data.py](./utils/process_data.py) functions for processing data

[training_nn.py](./utils/training_nn.py) functions for training neural network models

[evaluation.py](./utils/evaluation.py) functions for testing neural network models

[models.py](./utils/models.py) classes of architectures neural network models

[dataset.py](./utils/dataset.py) classes of Pytorch datasets

**Notebooks:**

[clas_ML.ipynb](./notebooks/clas_ML.ipynb) EDA, training boosting classifiers for antibody escape prediction by REGN33.

[DL_full_df.ipynb](./notebooks/DL_full_df.ipynb) training neural networks with full data of REGN33.

[limit_data.ipynb](./notebooks/limit_data.ipynb) training neural networks with limited data of REGN33.

[CNN_tr_learn.ipynb](./notebooks/CNN_tr_learn.ipynb) fine-tuning of CNN model using limited data of REGN33.

[RNN_tr_learn.ipynb](./notebooks/RNN_tr_learn.ipynb) fine-tuning of RNN model using limited data of REGN33. 

**Results:**

An RNN model typically consists of two main components: the recurrent block and the linear block.
The recurrent block is composed of identical architecture cells that take the current token in the sequence and an information vector from previous cells as input. This allows the RNN to process each amino acid in the protein sequence sequentially. The output of the recurrent block provides a useful representation of the entire RBD sequence, which is then fed into the linear block for classifying antibody escape.

![image](https://github.com/GavrilenkoA/ML_mutational_learning/assets/92908421/c49e34b6-6d58-49d5-be58-638f0b49347a)



Training RNN model consists of two stages:
1. Pre-training using background antibododies. Freeze weights of recurrent block.
2. Fine tuning using limited data of REGN33 antibody.

RNN model performance using combined data.


![image](https://github.com/GavrilenkoA/ML_mutational_learning/assets/92908421/a9dee58e-127f-4e87-afcf-511fc40ff9cf)



Then I provided 40 experiments, sampling different random 100 RBD sequences neutralized by REGN33 antibodies for fine-tuning and compared performance of base model and fine-tuning model, using paired T-test.

![image](https://github.com/GavrilenkoA/ML_mutational_learning/assets/92908421/8344ce73-e615-4f85-91b9-c0924f836953)

It can be seen, the fine-tuned model demonstrates more accurate and stable predictions.

#### Conclusion: 
Therapeutic antibodies share similar features - they target the conserved epitopes on RBD and mutations in the same manner reflected by binding RBD.
They can be used as a data source for pre-training ML models for prediction antibody escape with limited labeled data of antibody of interest.


