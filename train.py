import tensorflow as tf
import numpy as np
from transformer import build_transformer_model
from label_alignment import LABEL_CLASSES

def train_model(X, y, val_split=0.2, batch_size=4, epochs=50, lr=1e-4, label_smoothing=0.1):
    n_classes = len(LABEL_CLASSES)
    model = build_transformer_model(n_mels=X.shape[2], n_classes=n_classes)
    y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes)
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    # Handle single-file PoC: avoid empty splits
    if len(X) == 1:
        X_train, X_val = X, X
        y_train, y_val = y_cat, y_cat
        fit_epochs = min(epochs, 1)
    else:
        split = int(len(X)*(1-val_split))
        if split == 0:
            split = 1
        X_train, X_val = X[idx[:split]], X[idx[split:]]
        y_train, y_val = y_cat[idx[:split]], y_cat[idx[split:]]
        fit_epochs = epochs
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=['accuracy']
    )
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=fit_epochs, callbacks=cb)
    return model