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

# Constants
def get_container_root_folder(folder_name: str):
    return str((pathlib.Path(__file__).parent.parent / folder_name).resolve())

DATA_WAREHOUSE = get_container_root_folder('data_warehouse')
MODEL_REGISTRY = get_container_root_folder('model_registry')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:8080')

def train(**kwargs):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    input_size = kwargs.get('input_size', 8)
    hidden_size = kwargs.get('hidden_size', 128)
    num_layers = kwargs.get('num_layers', 3)
    output_size = kwargs.get('output_size', 2)
    sequence_length = kwargs.get('sequence_length', 30)
    train_ratio = kwargs.get('train_ratio', 0.8)
    ticker_name = kwargs.get('ticker_name', None)
    max_epochs = kwargs.get('max_epochs', 45)
    csv_path = os.path.join(DATA_WAREHOUSE, ticker_name + '_indicators.csv')

    full_dataset = StockDataset(csv_path, sequence_length=sequence_length)

    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    print(f"Dataset split for {ticker_name}: {train_size} training samples, {test_size} test samples")

    model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

    trainer = pl.Trainer(max_epochs=max_epochs)

    mlflow.pytorch.autolog()

    mlflow.set_experiment(experiment_name=f"{ticker_name} Prediction")

    run_name = f"{ticker_name}_h{hidden_size}_l{num_layers}_seq{sequence_length}_e{max_epochs}_historical"

    with mlflow.start_run(run_name=run_name):
        trainer.fit(model, train_dataloader)

        test_result = trainer.test(model, test_dataloader)
        print(f"Test results for {ticker_name}: {test_result}")
        
        if test_result and len(test_result) > 0:
            test_metrics = test_result[0]
            
            if 'test_f1_macro' in test_metrics:
                final_f1_macro = test_metrics['test_f1_macro']
                mlflow.log_metric("final_f1_macro", float(final_f1_macro))
                print(f"Final F1 Macro Score: {final_f1_macro}")
            
            if 'test_f1_weighted' in test_metrics:
                final_f1_weighted = test_metrics['test_f1_weighted']
                mlflow.log_metric("final_f1_weighted", float(final_f1_weighted))
                print(f"Final F1 Weighted Score: {final_f1_weighted}")
        
    test_result = trainer.test(model, test_dataloader)
    print(f"Test results for {ticker_name}: {test_result}")
            
            
            
    