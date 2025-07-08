import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn

class LSTMForecaster:
    """
    LSTM-based time series forecasting module that works with TimeSeries objects
    """
    def __init__(self, time_series, look_back=10, test_size=0.2, batch_size=8, device=torch.device("cpu")):
        """
        Args:
            time_series: TimeSeries object containing the data
            look_back: Number of previous time steps to use for prediction
            test_size: Proportion of data to use for testing (0-1)
            batch_size: Batch size for training
        """
        self.time_series = time_series
        self.look_back = look_back
        self.test_size = test_size
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.device = device
        self.model = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare the data for LSTM training"""
        # Use the values from TimeSeries object
        values = self.time_series.values

        # Handle different input types (pd.Series, np.array, etc.)
        if isinstance(values, pd.Series):
            values = values.values
        values = values.reshape(-1, 1)

        # Scale data
        self.scaler.fit(values[:int(len(values) * (1 - self.test_size))])  # Fit only on training data
        scaled_data = self.scaler.transform(values)

        # Split into train and test
        train_size = int(len(scaled_data) * (1 - self.test_size))
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - self.look_back:]  # Include look_back points

        # Create datasets
        self.train_dataset = TimeSeriesDataset(train_data, self.look_back)
        self.test_dataset = TimeSeriesDataset(test_data, self.look_back)

        # Create dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def init_model(self, input_size=1, hidden_sizes=[50], output_size=1, dropout=0.0):
        """Initialize the LSTM model"""
        self.model = ModeloLSTM(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            dropout=dropout
        ).to(self.device)

    def train(self, epochs=100, lr=0.001, verbose=True):
        """Train the LSTM model"""
        if self.model is None:
            raise ValueError("Model not initialized. Call init_model() first.")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss / len(self.train_loader.dataset):.6f}")

    def evaluate(self):
        """Evaluate the model on test data"""
        if self.model is None:
            raise ValueError("Model not initialized. Call init_model() first.")

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)

                # Inverse transform the scaled data
                pred = self.scaler.inverse_transform(pred.cpu().numpy())
                actual = self.scaler.inverse_transform(yb.cpu().numpy())

                predictions.append(pred[0][0])
                actuals.append(actual[0][0])

        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

        return predictions, actuals

    def forecast_future(self, steps=10):
        """Predice múltiples pasos hacia el futuro de forma autoregresiva"""
        if self.model is None:
            raise ValueError("Modelo no inicializado. Llama a init_model() primero.")

        self.model.eval()
        future_predictions = []

        # 1. Obtener la última secuencia de entrada [batch=1, seq_len, features=1]
        last_sequence = self.test_dataset.X[-1:].to(self.device)  # Shape: [1, look_back, 1]

        with torch.no_grad():
            for _ in range(steps):
                # 2. Hacer la predicción [1, 1]
                pred = self.model(last_sequence)

                # 3. Guardar la predicción (convertir a escalar)
                future_predictions.append(pred.item())

                # 4. Preparar la nueva entrada para el siguiente paso
                new_input = pred.unsqueeze(-1)

                # 5. Actualizar la secuencia:
                last_sequence = torch.cat([
                    last_sequence[:, 1:, :],  # [1, look_back-1, 1]
                    new_input  # [1, 1, 1]
                ], dim=1)  # Resultado: [1, look_back, 1]

        # 6. Convertir las predicciones a la escala original
        future_predictions = self.scaler.inverse_transform(
            np.array(future_predictions).reshape(-1, 1)
        ).flatten()

        return future_predictions

    def forecast_from_input(self, input_sequence, steps=10):
        """Predecir hacia el futuro dado un nuevo input"""
        if self.model is None:
            raise ValueError("Model not initialized. Call init_model() first.")

        self.model.eval()
        future_predictions = []

        # Convert input to tensor and shape [1, look_back, 1]
        if isinstance(input_sequence, (list, np.ndarray)):
            input_sequence = torch.FloatTensor(np.array(input_sequence).reshape(1, self.look_back, 1))
        elif isinstance(input_sequence, torch.Tensor):
            input_sequence = input_sequence.reshape(1, self.look_back, 1).float()
        else:
            raise ValueError("Unsupported input type for input_sequence")

        # Normalize using fitted scaler
        input_sequence = self.scaler.transform(input_sequence.cpu().numpy().reshape(-1, 1))
        last_sequence = torch.tensor(input_sequence.reshape(1, self.look_back, 1)).to(self.device)

        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(last_sequence)
                future_predictions.append(pred.item())

                new_input = pred.unsqueeze(-1)  # [1, 1, 1]
                last_sequence = torch.cat([last_sequence[:, 1:, :], new_input], dim=1)

        # Inverse transform the predictions
        return self.scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()



class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""

    def __init__(self, data, look_back=1):
        self.data = data
        self.look_back = look_back
        self.X, self.y = self._create_dataset()

    def _create_dataset(self):
        X, y = [], []
        for i in range(len(self.data) - self.look_back):
            X.append(self.data[i:i + self.look_back])
            y.append(self.data[i + self.look_back])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ModeloLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_sizes=[50], output_size=1, dropout=0.0):
        super(ModeloLSTM, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.num_layers = len(hidden_sizes)

        for ith_layer in range(self.num_layers):
            in_size = input_size if ith_layer == 0 else hidden_sizes[ith_layer - 1]
            out_size = hidden_sizes[ith_layer]
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=in_size,
                    hidden_size=out_size,
                    batch_first=True,
                    dropout=dropout if ith_layer < self.num_layers - 1 else 0.0
                )
            )

        self.linear = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        out = x
        for lstm in self.lstm_layers:
            out, _ = lstm(out)
        out = self.linear(out[:, -1, :])
        return out
