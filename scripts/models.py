import torch.nn as nn

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
    def __init__(self, input_size, seq_lentgth, hidden_size, num_classes=1):
        super(RNN, self).__init__()
        #self.dropout = nn.Dropout(0.25)
        self.hidden_size = hidden_size
        self.rnn1 = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn2 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.rnn3 = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.Linear1 = nn.Linear(seq_lentgth*hidden_size, 100)
        self.Linear2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x, h_1 = self.rnn1(x)
        x, h_2 = self.rnn2(x, h_1)
        x, _ = self.rnn3(x, h_2)
        x = x.reshape(x.size(0), -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        return x