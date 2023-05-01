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
* RNN
1. Trained only with data of antibody of interest
| Attempt | #1    | #2    |
| :-----: | :---: | :---: |
| Seconds | 301   | 283   |



