import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from hyperopt import Trials
import math
from warnings import simplefilter
import plotly.graph_objects as go
import time

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=DeprecationWarning)

# Streamlit app
def main():
    st.title("Forex Currency Price Prediction App")
    st.write("Ini adalah sistem prediksi yang menggunakan algoritma LSTM dengan input parameter. Hyperparameter adalah variabel yang secara signifikan mempengaruhi proses pelatihan model. Hyperparameter tidak dapat langsung diperoleh dari data dan perlu diinisialisasi sebelum proses pelatihan dimulai. Pada penelitian ini, inisialisasi hyperparameter meliputi pemilihan optimizer dan learning rate, penentuan batch size, jumlah epoch, dan struktur jaringan syaraf seperti jumlah hidden layer dan neuron pada setiap layer. Keputusan yang bijak dalam menentukan nilai-nilai tersebut dapat sangat mempengaruhi performa dan kemampuan generalisasi model, yang bertujuan untuk mencapai performa yang optimal dalam memprediksi harga mata uang forex")
    st.write("Automatic Parameters")

    st.header("Data Download")
    stock_symbol = st.selectbox("Enter Currency Pair:", ["JPY=X", "EURUSD=X", "AUDUSD=X", "IDR=X", "HKD=X", "BNDUSD=X"])
    start_date = st.date_input("Start Date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))
    price_type = st.selectbox("Select Price Type:", ["Close", "Open", "High", "Low"])

    data = yf.download(stock_symbol, start=start_date, end=end_date)

    close_prices = data[price_type].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    n_steps = 30
    X, y = prepare_data(scaled_data, n_steps)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    st.header("Hyperparameter Tuning")
    epoch_options = [50, 100, 150, 200]
    batch_size_options = [16, 32, 64, 128]
    epochs = epoch_options
    batch_size = batch_size_options

    if st.button("Apply Hyperparameters", key='apply_button'):
        st.sidebar.text("Applying hyperparameters...")
        space = {
            'units': 100,
            'dropout_rate': 0.2,  # Fixed value
            'learning_rate': 0.001,  # Fixed value
            'epochs': epochs,
            'batch_size': batch_size
        }

        results = []
        for e in epoch_options:
            for b in batch_size_options:
                st.write(f"Training with Epochs: {e}, Batch Size: {b}")
                start_time = time.time()

                best_params_lstm, history_lstm, y_pred_lstm, y_test_orig_lstm, model_summary = run_optimization({'units': 100, 'dropout_rate': 0.2, 'learning_rate': 0.001, 'epochs': e, 'batch_size': b}, 'lstm', X_train, y_train, X_test, y_test, scaler)

                end_time = time.time()
                duration = end_time - start_time

                st.write(f"Total time taken for prediction: {duration:.2f} seconds")

                mse, rmse, mape = display_results(history_lstm, y_test_orig_lstm, y_pred_lstm)

                results.append((e, b, mse, rmse, mape))

                visualize_predictions(data, train_size, n_steps, y_test_orig_lstm, y_pred_lstm)

        st.header("Summary of All Results")
        results_df = pd.DataFrame(results, columns=["Epochs", "Batch Size", "MSE", "RMSE", "MAPE"])
        st.write(results_df)

        best_params = results_df.loc[results_df['RMSE'].idxmin()]
        st.header("Best Parameters Based on RMSE")
        st.write(f"Epochs: {best_params['Epochs']}")
        st.write(f"Batch Size: {best_params['Batch Size']}")
        st.write(f"RMSE: {best_params['RMSE']}")

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)


def run_optimization(space, model_type, X_train, y_train, X_test, y_test, scaler):
    trials = Trials()

    best_params = space  # Directly use space as best_params since we are not using optimization
    final_model = build_model(best_params, X_train)

    history = final_model.fit(X_train, y_train,
                              epochs=best_params['epochs'],
                              batch_size=best_params['batch_size'],
                              verbose=2,
                              validation_split=0.1,)

    y_pred = final_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test_orig = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Get the model summary as a string
    stringlist = []
    final_model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)

    return best_params, history, y_pred, y_test_orig, model_summary


def build_model(params, X_train):
    model = Sequential()
    model.add(LSTM(units=params['units'], return_sequences=True, activation='tanh', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(params['dropout_rate']))
    model.add(LSTM(units=params['units'], activation='tanh'))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=params['learning_rate']), loss='mean_squared_error')
    return model


def display_results(history, y_test_orig, y_pred):
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_test_orig - y_pred) / y_test_orig)) * 100

    st.write("Mean Squared Error (MSE):", mse)
    st.write("Root Mean Squared Error (RMSE):", rmse)
    st.write("Mean Absolute Percentage Error (MAPE):", mape)

    st.line_chart(pd.DataFrame({'Train Loss': history.history['loss'], 'Validation Loss': history.history['val_loss']}))

    return mse, rmse, mape


def visualize_predictions(data, train_size, n_steps, y_test_orig, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index[:train_size + n_steps],
                             y=data['Close'].values[:train_size + n_steps],
                             mode='lines',
                             name="Training Data",
                             line=dict(color='gray')))

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_test_orig.flatten(),
                             mode='lines',
                             name="Actual Stock Prices",
                             line=dict(color='blue')))

    fig.add_trace(go.Scatter(x=data.index[train_size + n_steps:],
                             y=y_pred.flatten(),
                             mode='lines',
                             name="Predicted Stock Prices",
                             line=dict(color='red')))

    fig.update_layout(title="Forex Currency Price Prediction",
                      xaxis_title="Date",
                      yaxis_title="Price",
                      template='plotly_dark')

    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
