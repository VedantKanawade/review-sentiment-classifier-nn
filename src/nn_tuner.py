# src/nn_tuner.py

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def build_nn(input_dim, hidden_units=[128], dropout_rate=0.3, optimizer="adam", learning_rate=0.001):
    """
    Builds and returns a compiled feedforward neural network.
    """
    model = Sequential()
    # Input + first hidden layer
    model.add(Dense(hidden_units[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    # Additional hidden layers
    for units in hidden_units[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))
    
    # Output layer (binary classification)
    model.add(Dense(1, activation="sigmoid"))
    
    # Choose optimizer
    if optimizer.lower() == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer must be 'adam' or 'rmsprop'")
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def tune_nn_hyperparams(X_train, y_train, X_val, y_val, embedding_name="tfidf", epochs=20):
    """
    Searches over one hyperparameter at a time and plots validation accuracy/loss.
    Hyperparameters tested:
        - Hidden units: [128], [256], [128,64], [256,128], [512,256]
        - Dropout rate: [0.1,0.3,0.5,0.6]
        - Learning rate: [0.1,0.001,0.0005,0.0001]
        - Batch size: [16,32,64,128]
        - Optimizer: ['adam','rmsprop']
    """
    
    input_dim = X_train.shape[1]
    
    # Default params
    default_config = {
        "hidden_units": [128],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "adam"
    }
    
    # Hyperparameter grids
    hidden_units_list = [[128], [256], [128,64], [256,128], [512,256]]
    dropout_list = [0.1, 0.3, 0.5, 0.6]
    lr_list = [0.1, 0.001, 0.0005, 0.0001]
    batch_list = [16, 32, 64, 128]
    optimizer_list = ["adam", "rmsprop"]
    
    # Create folder for plots
    plot_folder = os.path.join("data", "plots", embedding_name)
    os.makedirs(plot_folder, exist_ok=True)
    
    # ---------- 1. Hidden units ----------
    val_accs = []
    val_losses = []
    for units in hidden_units_list:
        model = build_nn(input_dim=input_dim,
                         hidden_units=units,
                         dropout_rate=default_config["dropout_rate"],
                         optimizer=default_config["optimizer"],
                         learning_rate=default_config["learning_rate"])
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                            epochs=epochs, batch_size=default_config["batch_size"], verbose=0)
        val_accs.append(history.history["val_accuracy"][-1])
        val_losses.append(history.history["val_loss"][-1])
    
    # Plot
    plt.figure(figsize=(8,5))
    labels = ["[128]","[256]","[128,64]","[256,128]","[512,256]"]
    plt.plot(labels, val_accs, marker="o", label="Val Accuracy")
    plt.plot(labels, val_losses, marker="x", label="Val Loss")
    plt.title("Hidden Units Tuning")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "hidden_units_tuning.png"))
    plt.show()
    
    # ---------- 2. Dropout ----------
    val_accs, val_losses = [], []
    for rate in dropout_list:
        model = build_nn(input_dim=input_dim,
                         hidden_units=default_config["hidden_units"],
                         dropout_rate=rate,
                         optimizer=default_config["optimizer"],
                         learning_rate=default_config["learning_rate"])
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                            epochs=epochs, batch_size=default_config["batch_size"], verbose=0)
        val_accs.append(history.history["val_accuracy"][-1])
        val_losses.append(history.history["val_loss"][-1])
    
    plt.figure(figsize=(8,5))
    plt.plot(dropout_list, val_accs, marker="o", label="Val Accuracy")
    plt.plot(dropout_list, val_losses, marker="x", label="Val Loss")
    plt.title("Dropout Rate Tuning")
    plt.xlabel("Dropout Rate")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "dropout_tuning.png"))
    plt.show()
    
    # ---------- 3. Learning Rate ----------
    val_accs, val_losses = [], []
    for lr in lr_list:
        model = build_nn(input_dim=input_dim,
                         hidden_units=default_config["hidden_units"],
                         dropout_rate=default_config["dropout_rate"],
                         optimizer=default_config["optimizer"],
                         learning_rate=lr)
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                            epochs=epochs, batch_size=default_config["batch_size"], verbose=0)
        val_accs.append(history.history["val_accuracy"][-1])
        val_losses.append(history.history["val_loss"][-1])
    
    plt.figure(figsize=(8,5))
    plt.plot([str(lr) for lr in lr_list], val_accs, marker="o", label="Val Accuracy")
    plt.plot([str(lr) for lr in lr_list], val_losses, marker="x", label="Val Loss")
    plt.title("Learning Rate Tuning")
    plt.xlabel("Learning Rate")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "learning_rate_tuning.png"))
    plt.show()
    
    # ---------- 4. Batch Size ----------
    val_accs, val_losses = [], []
    for batch in batch_list:
        model = build_nn(input_dim=input_dim,
                         hidden_units=default_config["hidden_units"],
                         dropout_rate=default_config["dropout_rate"],
                         optimizer=default_config["optimizer"],
                         learning_rate=default_config["learning_rate"])
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                            epochs=epochs, batch_size=batch, verbose=0)
        val_accs.append(history.history["val_accuracy"][-1])
        val_losses.append(history.history["val_loss"][-1])
    
    plt.figure(figsize=(8,5))
    plt.plot(batch_list, val_accs, marker="o", label="Val Accuracy")
    plt.plot(batch_list, val_losses, marker="x", label="Val Loss")
    plt.title("Batch Size Tuning")
    plt.xlabel("Batch Size")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "batch_size_tuning.png"))
    plt.show()
    
    # ---------- 5. Optimizer ----------
    val_accs, val_losses = [], []
    for opt in optimizer_list:
        model = build_nn(input_dim=input_dim,
                         hidden_units=default_config["hidden_units"],
                         dropout_rate=default_config["dropout_rate"],
                         optimizer=opt,
                         learning_rate=default_config["learning_rate"])
        history = model.fit(X_train, y_train, validation_data=(X_val,y_val),
                            epochs=epochs, batch_size=default_config["batch_size"], verbose=0)
        val_accs.append(history.history["val_accuracy"][-1])
        val_losses.append(history.history["val_loss"][-1])
    
    plt.figure(figsize=(8,5))
    plt.plot(optimizer_list, val_accs, marker="o", label="Val Accuracy")
    plt.plot(optimizer_list, val_losses, marker="x", label="Val Loss")
    plt.title("Optimizer Tuning")
    plt.ylabel("Metric")
    plt.legend()
    plt.savefig(os.path.join(plot_folder, "optimizer_tuning.png"))
    plt.show()
    
    print("All hyperparameter tuning plots saved in:", plot_folder)
