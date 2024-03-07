import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


def encode_normalize(df, train_start, train_end, test_end, selected_vars=None):

    train_start = str(train_start)
    train_end = str(train_end)
    test_end = str(test_end)

    # Variable definitions
    if type(selected_vars) != type(None):
        X_vars = selected_vars.copy()
    else:
        X_vars = [item for item in list(df.columns) if item not in ['Date', 'Emissions']]

    add_vars = [
        "Day_sin",
        "Day_cos",
        "Month_sin",
        "Month_cos",
        "Year_sin",
        "Year_cos",
        ]

    X_vars += add_vars
    y_vars = ["Emissions"]

    cat_vars = ["Region", "saison", "week_day", "is_holiday", "is_bank_holiday"]
    scalar_vars = [item for item in (X_vars + y_vars) if item not in cat_vars]

    # Load data
    data = df.copy()

    # Encode categorical variables
    encoder = LabelEncoder()
    for var in cat_vars:
        data[var] = encoder.fit_transform(data[var])

    # Convert Date to sinus and cosinus transform
    data["Date"] = pd.to_datetime(data["Date"])
    data["Day_sin"] = np.sin(2 * np.pi * data["Date"].dt.day / 31)
    data["Day_cos"] = np.cos(2 * np.pi * data["Date"].dt.day / 31)
    data["Month_sin"] = np.sin(2 * np.pi * data["Date"].dt.month / 12)
    data["Month_cos"] = np.cos(2 * np.pi * data["Date"].dt.month / 12)
    data["Year_sin"] = np.sin(2 * np.pi * data["Date"].dt.year)
    data["Year_cos"] = np.cos(2 * np.pi * data["Date"].dt.year)

    # Normalize numerical features
    scaler = StandardScaler()
    data[scalar_vars] = scaler.fit_transform(data[scalar_vars])

    # Split data into features and target
    X = data[X_vars]
    y = data[y_vars]

    # Split data into training and testing sets
    is_train = (data['Date'] >= train_start) * (data['Date'] <= train_end)
    is_test = (data['Date'] > train_end) * (data['Date'] <= test_end)
    X_train, X_test = X[is_train], X[is_test]
    y_train, y_test = y[is_train], y[is_test]

    return X_train, X_test, y_train, y_test


def LSTM_predict(df, train_start, train_end, test_end, selected_vars=None):

    X_train, X_test, y_train, y_test = encode_normalize(df, train_start, train_end, test_end, selected_vars)

    # Reshape features for LSTM input [samples, timesteps, features]
    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test.values, (X_test.shape[0], 1, X_test.shape[1]))

    # Define the LSTM model
    model = Sequential(
        [
            LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss="mse")

    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.5,
        callbacks=[early_stopping],
    )

    # Predict on test data
    y_pred = model.predict(X_test).ravel()

    # Compute final dataframe
    y_frame = pd.DataFrame({"y" : np.concatenate((np.array(y_train).ravel(),np.array(y_test).ravel()))})
    y_pred_frame = pd.DataFrame({"y_pred" : y_pred}).set_index(pd.DataFrame(y_test).index)
    output_df = df[['Date','Region']].join(pd.concat((y_frame,y_pred_frame),axis=1))

    return output_df


def Catboost_predict(df, train_start, train_end, test_end, selected_vars=None):

    X_train, X_test, y_train, y_test = encode_normalize(df, train_start, train_end, test_end, selected_vars)

    # Define CatBoost model
    model = CatBoostRegressor(
        iterations=500, learning_rate=0.5, depth=6, loss_function="RMSE"
    )

    # Train CatBoost model
    model.fit(
        X_train,
        y_train,
        verbose=100,
        early_stopping_rounds=50,
    )

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute final dataframe
    y_frame = pd.DataFrame({"y" : np.concatenate((np.array(y_train).ravel(),np.array(y_test).ravel()))})
    y_pred_frame = pd.DataFrame({"y_pred" : y_pred}).set_index(pd.DataFrame(y_test).index)
    output_df = df[['Date','Region']].join(pd.concat((y_frame,y_pred_frame),axis=1))

    return output_df


def Xgboost_predict(df, train_start, train_end, test_end, selected_vars=None):

    X_train, X_test, y_train, y_test = encode_normalize(df, train_start, train_end, test_end, selected_vars)

    # Train Xgboost model
    model = XGBRegressor(objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute final dataframe
    y_frame = pd.DataFrame({"y" : np.concatenate((np.array(y_train).ravel(),np.array(y_test).ravel()))})
    y_pred_frame = pd.DataFrame({"y_pred" : y_pred}).set_index(pd.DataFrame(y_test).index)
    output_df = df[['Date','Region']].join(pd.concat((y_frame,y_pred_frame),axis=1))

    return output_df