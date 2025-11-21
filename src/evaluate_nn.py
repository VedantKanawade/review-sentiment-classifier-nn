import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def build_nn(input_dim, hidden_units=[128], dropout_rate=0.3, optimizer="adam", learning_rate=0.001):
    """Builds and compiles a feedforward neural network for binary classification"""
    model = Sequential()
    model.add(Dense(hidden_units[0], activation="relu", input_shape=(input_dim,)))
    model.add(Dropout(dropout_rate))
    
    for units in hidden_units[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation="sigmoid"))
    
    if optimizer.lower() == "adam":
        opt = Adam(learning_rate=learning_rate)
    elif optimizer.lower() == "rmsprop":
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer must be 'adam' or 'rmsprop'")
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

def evaluate_nn(X_train, y_train, X_val, y_val, X_test, y_test,
                hidden_units, dropout_rate, learning_rate,
                batch_size, optimizer, embedding_name="embedding", epochs=30):
    """
    Trains NN on train+val, evaluates on test, prints metrics, plots confusion matrix
    """
    
    # Combine train + val
    X_combined = np.concatenate([X_train, X_val], axis=0)
    y_combined = np.concatenate([y_train, y_val], axis=0)
    
    input_dim = X_combined.shape[1]
    
    # Build model
    model = build_nn(input_dim=input_dim,
                     hidden_units=hidden_units,
                     dropout_rate=dropout_rate,
                     optimizer=optimizer,
                     learning_rate=learning_rate)
    
    # Train
    history = model.fit(X_combined, y_combined,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=1)
    
    # Predict on test
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {embedding_name}")
    plt.show()
    
    return model, history
