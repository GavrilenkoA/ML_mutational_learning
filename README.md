### Antibody escape prediction of mutated SARS-CoV-2 variants.
***
#### Purpose and objectives of the project

##### Purpose:
* Develop an algorithm which can perform better in condition with few data mutated variants using fine-tuning technique.
##### Objectives:
* Reproduce the article https://doi.org/10.1101/2021.12.07.471580 - classification antibody escape (only LY15 antibody) using random forest and LSTM models.
* Merge the data of mutated SARS-CoV-2 variants with four antibodies into one dataframe.
* Pretrain models using background antibodies and fine-tuning on few data of antibody of interest (REGN33). Compare metrics with model performance trained only on few data of antibody of interest.

##### Results:
In the course of this project two models were designed:
1. RNN model
Belonging mutated variants to specific antibodies was encoded as additional embedding.

a) Trained only with data of antibody of interest (REGN33 - 100 sequences). 

| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.756083      |  0.695548     |0.790864       |0.74015        | 0.838347      |

b) Fine-tuning pretrained model with data of background antibodies. 


| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
|0.819495       |  0.890721     | 0.790864      |0.83134        | 0.903893      |

2. 1D CNN model
Belonging mutated variants to specific antibodies was as ones in corresponding position in vector.

a) Trained only with data of antibody of interest (REGN33 - 100 sequences). 

| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.767463      |   0.726878    |0.790641       | 0.75742       | 0.853307      |

b) Fine-tuning pretrained model with data of background antibodies. 

| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.724115      |   0.90395     | 0.644545      |  0.861503     | 0.861503      |

RNN model was selected finally. 
Following distribution ROC-AUC metrics with 40 samples of REGN33 data. 
![alt text]


