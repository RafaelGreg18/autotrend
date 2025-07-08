import torch
import torch.nn as nn
import torchmetrics
import torchmetrics.classification
import pytorch_lightning as pl

class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, lr=0.001):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        
        self.lr = lr
        

        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.train_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=2, average="macro")
        self.train_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=2, average="macro")
        self.train_f1_macro = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average="macro")
        self.train_f1_weighted = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average="weighted")
        self.train_f1_per_class = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average=None)
        
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=2)
        self.test_precision = torchmetrics.classification.Precision(task="multiclass", num_classes=2, average="macro")
        self.test_recall = torchmetrics.classification.Recall(task="multiclass", num_classes=2, average="macro")
        self.test_f1_macro = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average="macro")
        self.test_f1_weighted = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average="weighted")
        self.test_f1_per_class = torchmetrics.classification.F1Score(task="multiclass", num_classes=2, average=None)

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
        
        acc = self.train_accuracy(preds, y)
        prec = self.train_precision(preds, y)
        rec = self.train_recall(preds, y)
        f1_macro = self.train_f1_macro(preds, y)
        f1_weighted = self.train_f1_weighted(preds, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_precision', prec)
        self.log('train_recall', rec)
        self.log('train_f1_macro', f1_macro)
        self.log('train_f1_weighted', f1_weighted)

        f1_per_class = self.train_f1_per_class(preds, y)
        for i, f1_score in enumerate(f1_per_class):
            self.log(f'train_f1_class_{i}', f1_score)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        

        preds = torch.argmax(y_hat, dim=1)
        

        acc = self.test_accuracy(preds, y)
        prec = self.test_precision(preds, y)
        rec = self.test_recall(preds, y)
        f1_macro = self.test_f1_macro(preds, y)
        f1_weighted = self.test_f1_weighted(preds, y)
        

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_precision', prec)
        self.log('test_recall', rec)
        self.log('test_f1_macro', f1_macro)
        self.log('test_f1_weighted', f1_weighted)
        

        f1_per_class = self.test_f1_per_class(preds, y)
        for i, f1_score in enumerate(f1_per_class):
            self.log(f'test_f1_class_{i}', f1_score)

        return {
            'test_loss': loss, 
            'test_f1_macro': f1_macro, 
            'test_f1_weighted': f1_weighted,
            'test_acc': acc
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)