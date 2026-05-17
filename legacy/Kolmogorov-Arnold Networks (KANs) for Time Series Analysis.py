"""Generated from Jupyter notebook: Kolmogorov-Arnold Networks (KANs) for Time Series Analysis

Magics and shell lines are commented out. Run with a normal Python interpreter."""

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from kan import KAN
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class KolmogorovArnoldNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.decomposition = nn.Linear(input_dim, hidden_dim)
        self.aggregation = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.decomposition(x))
        g = self.aggregation(h)
        return g


class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


def fetch_fred_data(series_id, api_key, start_date="2000-01-01", save_csv=False):
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": datetime.now().strftime("%Y-%m-%d"),
    }
    url = "https://api.stlouisfed.org/fred/series/observations"
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        observations = data["observations"]
        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()
        df = df.sort_values("date")
        df = df.set_index("date")
        if save_csv:
            csv_filename = f"{series_id}_data.csv"
            df.to_csv(csv_filename)
            print(f"Data saved to {csv_filename}")
        return df
    else:
        raise Exception(f"API request failed with status code {response.status_code}")


def main() -> None:
    np.random.seed(42)

    time = np.linspace(0, 10, 500)

    values = 10 + 2 * np.sin(time) + 0.5 * np.random.normal(size=len(time))

    df = pd.DataFrame({"time": time, "value": values})

    plt.figure(figsize=(10, 6))

    plt.plot(df["time"], df["value"], label="Synthetic Time Series")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.title("Synthetic Time Series Data")

    plt.legend()

    plt.show()

    np.random.seed(42)

    time = np.linspace(0, 10, 500)

    values = 10 + 2 * np.sin(time) + 0.5 * np.random.normal(size=len(time))

    df = pd.DataFrame({"time": time, "value": values})

    plt.figure(figsize=(10, 6))

    plt.plot(df["time"], df["value"], label="Synthetic Time Series")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.title("Synthetic Time Series Data")

    plt.legend()

    plt.show()

    X = np.array(df["time"]).reshape(-1, 1)

    y = np.array(df["value"]).reshape(-1, 1)

    scaler_X = MinMaxScaler()

    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)

    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    input_dim = 1

    hidden_dim = 10

    output_dim = 1

    learning_rate = 0.01

    num_epochs = 100

    model = KolmogorovArnoldNetwork(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = scaler_y.inverse_transform(predictions.numpy())
        y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test_actual, predictions, alpha=0.7)

    plt.plot(
        [min(y_test_actual), max(y_test_actual)],
        [min(y_test_actual), max(y_test_actual)],
        color="red",
    )

    plt.xlabel("Actual Values")

    plt.ylabel("Predicted Values")

    plt.title("KAN Predictions vs Actual")

    plt.grid()

    plt.show()

    np.random.seed(42)

    time = np.linspace(0, 10, 500)

    values = 10 + 2 * np.sin(time) + 0.5 * np.random.normal(size=len(time))

    df = pd.DataFrame({"time": time, "value": values})

    plt.figure(figsize=(10, 6))

    plt.plot(df["time"], df["value"], label="Synthetic Time Series")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.title("Synthetic Time Series Data")

    plt.legend()

    plt.show()

    X = np.array(df["time"]).reshape(-1, 1)

    y = np.array(df["value"]).reshape(-1, 1)

    scaler_X = MinMaxScaler()

    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)

    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    input_dim = 1

    hidden_dim = 10

    output_dim = 1

    learning_rate = 0.01

    num_epochs = 100

    model = KolmogorovArnoldNetwork(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = scaler_y.inverse_transform(predictions.numpy())
        y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test_actual, predictions, alpha=0.7)

    plt.plot(
        [min(y_test_actual), max(y_test_actual)],
        [min(y_test_actual), max(y_test_actual)],
        color="red",
    )

    plt.xlabel("Actual Values")

    plt.ylabel("Predicted Values")

    plt.title("KAN Predictions vs Actual")

    plt.grid()

    plt.show()

    X = df.drop("value", axis=1)

    y = df["value"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    model = KAN(width=[X_train.shape[1], 10, 1], grid=5, k=3)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    np.random.seed(42)

    time = np.linspace(0, 10, 500)

    values = 10 + 2 * np.sin(time) + 0.5 * np.random.normal(size=len(time))

    df = pd.DataFrame({"time": time, "value": values})

    plt.figure(figsize=(10, 6))

    plt.plot(df["time"], df["value"], label="Synthetic Time Series")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.title("Synthetic Time Series Data")

    plt.legend()

    plt.show()

    X = np.array(df["time"]).reshape(-1, 1)

    y = np.array(df["value"]).reshape(-1, 1)

    scaler_X = MinMaxScaler()

    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)

    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    input_dim = 1

    hidden_dim = 10

    output_dim = 1

    learning_rate = 0.01

    num_epochs = 100

    model = SimpleNN(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = scaler_y.inverse_transform(predictions.numpy())
        y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test_actual, predictions, alpha=0.7)

    plt.plot(
        [min(y_test_actual), max(y_test_actual)],
        [min(y_test_actual), max(y_test_actual)],
        color="red",
    )

    plt.xlabel("Actual Values")

    plt.ylabel("Predicted Values")

    plt.title("Neural Network Predictions vs Actual")

    plt.grid()

    plt.show()

    mse = np.mean((predictions - y_test_actual) ** 2)

    print(f"Mean Squared Error: {mse:.4f}")

    api_key = "8f058d10ec8c788296c040ea09e634d5"

    series_id = "T10Y2Y"

    df = fetch_fred_data(series_id, api_key)

    print("Data fetched successfully.")

    df["time"] = (df.index - df.index[0]).days

    X = df["time"].values.reshape(-1, 1)

    y = df["value"].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()

    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(X)

    y = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    input_dim = 1

    hidden_dim = 10

    output_dim = 1

    learning_rate = 0.01

    num_epochs = 100

    model = SimpleNN(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()

    with torch.no_grad():
        predictions = model(X_test_tensor)
        predictions = scaler_y.inverse_transform(predictions.numpy())
        y_test_actual = scaler_y.inverse_transform(y_test_tensor.numpy())

    plt.figure(figsize=(10, 6))

    plt.scatter(y_test_actual, predictions, alpha=0.7)

    plt.plot(
        [min(y_test_actual), max(y_test_actual)],
        [min(y_test_actual), max(y_test_actual)],
        color="red",
    )

    plt.xlabel("Actual Values")

    plt.ylabel("Predicted Values")

    plt.title("Neural Network Predictions vs Actual")

    plt.grid()

    plt.savefig("NN_Predictions_vs_Actual.png")

    plt.show()

    mse = np.mean((predictions - y_test_actual) ** 2)

    print(f"Mean Squared Error: {mse:.4f}")

    plt.figure(figsize=(12, 6))

    plt.plot(df.index, scaler_y.inverse_transform(y), label="Original Data", alpha=0.7)

    plt.scatter(
        df.index[int(0.8 * len(df)) :], predictions, color="red", label="NN Predictions", alpha=0.7
    )

    plt.xlabel("Date")

    plt.ylabel("Value")

    plt.title("Time Series with Neural Network Predictions")

    plt.legend()

    plt.savefig("NN_Time_Series_Predictions.png")

    plt.show()

    print("Forecasting completed and visualizations saved.")


if __name__ == "__main__":
    main()
