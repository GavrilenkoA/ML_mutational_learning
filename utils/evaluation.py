import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_loss(train_loss: list, valid_loss: list) -> None:
    loss_df = pd.DataFrame(list(zip(train_loss, valid_loss)), columns=['train_loss', 'valid_loss'])
    sns.lineplot(data=loss_df)
    plt.show()


def measure_metrics(y_true: np.ndarray, y_pred: np.ndarray, pred_logits: np.ndarray) -> list[float]:
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1_scor = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, pred_logits)
    return list((accuracy, recall, precision, f1_scor, auc))


def evaluate_model(loader: DataLoader, model: nn.Module) -> pd.DataFrame:
    m = nn.Sigmoid()
    model.eval()
    ground_truth = []
    all_prob = []
    all_predict = []
    with torch.no_grad():
        for batch in loader:
            input_embeds, labels = batch
            input_embeds = input_embeds.float().to(device)
            labels = labels.unsqueeze(1).to(device)
            labels = labels.float()
            prediction = model(input_embeds)
            prediction = prediction.squeeze(1)
            pred = torch.where(prediction > 0.5, 1, 0)
            all_predict.append(pred.cpu().numpy())
            prob = m(prediction)
            all_prob.append(prob.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())
    all_prob = np.concatenate(all_prob)
    ground_truth = np.concatenate(ground_truth)
    all_predict = np.concatenate(all_predict)
    metrics = measure_metrics(ground_truth, all_predict, all_prob)
    columns = ['accuracy', 'recall', 'precision', 'f1_scor', 'auc']
    metrics = pd.DataFrame(dict(zip(columns, metrics)), index=[0])
    return metrics


def evaluate_model_rnn(loader: DataLoader, model: nn.Module) -> pd.DataFrame:
    m = nn.Sigmoid()
    model.eval()
    ground_truth = []
    all_prob = []
    all_predict = []
    with torch.no_grad():
        for batch in loader:
            feature, ab, labels = batch
            ab = ab.to(device)
            feature = feature.float().to(device)
            labels = labels.unsqueeze(1).to(device).float()
            prediction = model(feature, ab)
            prediction = prediction.squeeze(1)
            pred = torch.where(prediction > 0.5, 1, 0)
            all_predict.append(pred.cpu().numpy())
            prob = m(prediction)
            all_prob.append(prob.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())
    all_prob = np.concatenate(all_prob)
    ground_truth = np.concatenate(ground_truth)
    all_predict = np.concatenate(all_predict)
    metrics = measure_metrics(ground_truth, all_predict, all_prob)
    columns = ['accuracy', 'recall', 'precision', 'f1_scor', 'auc']
    metrics = pd.DataFrame(dict(zip(columns, metrics)), index=[0])
    return metrics
