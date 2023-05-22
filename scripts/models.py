
import torch.nn as nn
import torch
import torch.nn.functional as F



class CNNModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_classes=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(embed_size, hidden_size, kernel_size=3,
                      padding=1, stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, 
                      padding=1, stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, 
                      stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(), nn.ReLU()
        )
        self.cl = nn.Sequential(
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        prediction = self.cl(x)
        return prediction


class RNN(nn.Module):
    def __init__(self, input_size=41, seq_lentgth=24, hidden_size=140, num_layers=10, num_classes=1):
        super().__init__()
        self.embed = nn.Embedding(5, seq_lentgth, dtype=torch.float32)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.linear1 = nn.Linear(2*seq_lentgth*hidden_size, 100)
        self.linear2 = nn.Linear(100, num_classes)

    def forward(self, x, ab_idx):
        embed = self.embed(ab_idx)
        embed = torch.unsqueeze(embed, 2)
        embed = torch.cat((embed, x), dim=2)
        x, _ = self.rnn(embed)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
