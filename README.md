### Antibody escape prediction of mutated SARS-CoV-2 variants.
***
#### Purpose and objectives of the project

##### Purpose:
* Develop an algorithm which can perform better in condition with few data of mutated variants RBD using fine-tuning technique.
##### Objectives:
* Reproduce the article https://doi.org/10.1101/2021.12.07.471580 - classification antibody escape (only LY16 antibody data) using CatBoost and RandomForest models.
* Add physical features to RBD mutation protein sequences and merge the data of mutated SARS-CoV-2 variants neutralyzing by four antibodies into one dataset.
* Train models process sequential data: Recurrent and Convolution neural networks.
* Pretrain models using background antibodies and fine-tuning on few data of antibody of interest (REGN33). Compare metrics with model performance trained only on few data of antibody of interest.

### Results:
#### A) Learning classical ML models using data one antibody separately.
##### A.1) Escape prediction by LY16 antibody training Random forest model.
.

| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.883495    |   0.886102      |   0.878951   |0.882512       | 0.94634      |

##### A.2) Escape prediction by LY16 antibody training Catboost model on ACE2 data.
.
| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.495815    |  0.970169    |0.494642      |0.65522      | 0.551635   |
As a result of recall metrics ACE2 binding data could relevant for ly16 escape prediction, almost absent false negative prediction.

#### B) In the course of this project two neural network models were designed: RNN и CNN.
##### B.1) Escape prediction by LY555 antibody training CNN model.
.
| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.902803  |  0.852246    |0.949934     |0.898442     | 0.961902 |
##### B.2) Escape prediction by LY555 antibody training RNN model.
.
| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.878951  |  0.853428    |0.901373    |0.876746    | 0.942996 |

##### B.3) Escape prediction by LY555 antibody training CNN model using physical features.
.
| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.902206  |  0.847518   |0.953457   |0.897372    | 0.959425 |

##### B.4) Escape prediction by LY555 antibody training RNN model using physical features.
.
| Accuracy      | Recall        | Precision     | F1-Score      | ROC-AUC       |   
| ------------- | ------------- | ------------- | --------------| --------------|
| 0.905188  |  0.868794    |0.938697   |0.902394    | 0.957798 |
Adding physical features increase performance RNN models.





#### C) Training DL model with limited labeled train data.
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
![image](https://user-images.githubusercontent.com/92908421/235514681-867064d8-9a41-4c0f-82be-58c550dbe373.png)

#### Workflow overview
##### Notebooks

* single_onehot.ipynb - LY16 escape prediction training Catboost and Random Forest models.
* DL_full_df.ipynb - LY555 escape prediction training RNN and CNN models.
* limit_data.ipynb - REGN33 escape prediction using limit train data using physical features.
* cnn_1d.ipynb - REGN33 escape prediction train CNN using fine-tuning technique.
* recurrent-model.ipynb - REGN33 escape prediction train RNN using fine-tuning technique.



