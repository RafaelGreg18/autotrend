import pytorch_lightning as pl
import torch
import torchmetrics
import torch.nn as nn

class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, lr=0.001):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        
        self.lr = lr
        
        self.accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.precision = torchmetrics.classification.Precision(task="multiclass", num_classes=2, average="macro")
        self.recall = torchmetrics.classification.Recall(task="multiclass", num_classes=2, average="macro")

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        
        if lstm_out.shape[1] == 1:
            out = lstm_out.squeeze(1)
        else:
            out = lstm_out[:, -1, :]
        
        out = self.dropout(out)
        out = self.fc(out)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_precision', prec)
        self.log('train_recall', rec)

        return loss

    def test_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', accuracy)
        self.log('test_precision', self.precision(predicted, y))
        self.log('test_recall', self.recall(predicted, y))
        
        return {'test_loss': loss, 'test_acc': accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)