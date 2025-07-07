# Libraries
import pytorch_lightning as pl
import torch
import mlflow
import mlflow.pytorch

# Custom dataset and model
from stock_dataset import StockDataset
from lstm_classifier import LSTMClassifier

# System libraries
import pathlib
import os

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM model for stock prediction.")

    parser.add_argument('--set_tracking_uri', type=str, default='http://127.0.0.1:8080')

    return parser.parse_args()

def train(**kwargs):
    ## TODO: continue transcribing the training process from main
    input_size = kwargs.get('input_size', 8)
    hidden_size = kwargs.get('hidden_size', 128)
    num_layers = kwargs.get('num_layers', 3)
    output_size = kwargs.get('output_size', 2)
    sequence_length = kwargs.get('sequence_length', 30)
    train_ratio = kwargs.get('train_ratio', 0.8)
    ticker_name = kwargs.get('ticker_name', None)
    csv_path = kwargs.get('csv_path', None)

    # Load the full dataset with sequence length
    full_dataset = StockDataset(csv_path, sequence_length=sequence_length)

    # Calculate the split sizes
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    # Split the dataset
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    # Create data loaders for both sets
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    print(f"Dataset split for {ticker_name}: {train_size} training samples, {test_size} test samples")

    # Initialize the model
    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

    # Initialize the trainer with validation data
    trainer = pl.Trainer(max_epochs=max_epochs)

    mlflow.pytorch.autolog() # This is necessary to log the model and metrics automatically

    mlflow.set_experiment(experiment_name=f"{ticker_name} Prediction")

    run_name = f"{ticker_name}_h{hidden_size}_l{num_layers}_seq{sequence_length}_e{max_epochs}_historical"

    # Train the model
    with mlflow.start_run(run_name=run_name) as run:
        trainer.fit(model, train_dataloader)

    # Evaluate on test data
    test_result = trainer.test(model, test_dataloader)
    print(f"Test results for {ticker_name}: {test_result}")
    
if __name__ == "__main__":
    args = parse_args()
    mlflow.set_tracking_uri(args.set_tracking_uri)

    ## TODO: retrieve these parameters from a config file
    input_size = 8  # Number of features
    hidden_size = 64  # Number of LSTM units
    num_layers = 3  # Number of LSTM layers
    output_size = 2  # Number of classes
    sequence_length = 30  # Number of time steps in each sequence
    max_epochs = 45 # Number of epochs for training
    # Define train/test split ratio
    train_ratio = 0.8  # 80% for training, 20% for testing
    
    data_warehouse = (pathlib.Path(__file__).parent.parent / 'data_warehouse').resolve()
    data_warehouse = str(data_warehouse)
    for data in os.listdir(data_warehouse):
        if data.endswith('_indicators.csv'):
            ticker_name = data.removesuffix('_indicators.csv')
            csv_path = os.path.join(data_warehouse, data)
            train(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=output_size,
                sequence_length=sequence_length,
                max_epochs=max_epochs,
                train_ratio=train_ratio,
                ticker_name=ticker_name,
                csv_path=csv_path
            )
    print("Training completed for all datasets.")
    print("All models trained and logged successfully.")
            
            
            
    