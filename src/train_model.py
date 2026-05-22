"""
CryptoCast - Train a single model for a single horizon
=======================================================
Useful for quick experiments or retraining a specific model.

Usage:
    python src/train_model.py <model_name> <horizon_name>
    E.g.:  python src/train_model.py 1D-CNN 1D
"""
import os, sys, warnings, json, pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, SimpleRNN, Conv1D,
                                      GlobalAveragePooling1D, Dropout,
                                      Input, MultiHeadAttention, LayerNormalization, Add)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

model_name = sys.argv[1]
horizon_name = sys.argv[2]
EPOCHS = 50
BATCH_SIZE = 64
SEQ_LENGTH = 60
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

horizon_map = {'1D': 1, '3D': 3, '7D': 7}
horizon = horizon_map[horizon_name]

# Load data
scaled_data = np.load(os.path.join(PROJECT_DIR, 'scaled_data.npy'))
with open(os.path.join(PROJECT_DIR, 'scalers.pkl'), 'rb') as f:
    scalers = pickle.load(f)
target_scaler = scalers['target_scaler']

with open(os.path.join(PROJECT_DIR, 'meta.json'), 'r') as f:
    meta = json.load(f)
TEST_RATIO = meta['test_ratio']

def create_sequences(data_array, seq_length, horizon, target_idx=0):
    X, y = [], []
    for i in range(len(data_array) - seq_length - horizon + 1):
        X.append(data_array[i:i + seq_length])
        y.append(data_array[i + seq_length + horizon - 1, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, SEQ_LENGTH, horizon, target_idx=0)
split_seq = int(len(X) * (1 - TEST_RATIO))
X_train, X_test = X[:split_seq], X[split_seq:]
y_train, y_test = y[:split_seq], y[split_seq:]
input_shape = (X_train.shape[1], X_train.shape[2])

print(f"Training {model_name} for {horizon_name} horizon...")
print(f"  X_train={X_train.shape}, X_test={X_test.shape}")

def build_cnn(input_shape):
    return Sequential([
        Conv1D(64, 3, activation='relu', padding='causal', input_shape=input_shape),
        Conv1D(64, 3, activation='relu', padding='causal'),
        Conv1D(32, 3, activation='relu', padding='causal'),
        GlobalAveragePooling1D(), Dropout(0.2),
        Dense(64, activation='relu'), Dropout(0.2), Dense(1)
    ], name='CNN_1D')

def build_rnn(input_shape):
    return Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), SimpleRNN(32), Dropout(0.2),
        Dense(32, activation='relu'), Dense(1)
    ], name='RNN')

def build_lstm(input_shape):
    return Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2), LSTM(64, return_sequences=True),
        Dropout(0.2), LSTM(32), Dropout(0.2),
        Dense(64, activation='relu'), Dense(1)
    ], name='LSTM')

def build_transformer(input_shape, head_size=64, num_heads=4, ff_dim=128, num_blocks=2):
    inputs = Input(shape=input_shape)
    x = Dense(head_size * num_heads)(inputs)
    for _ in range(num_blocks):
        attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=0.1)(x, x)
        attn = Dropout(0.1)(attn)
        x1 = Add()([x, attn])
        x1 = LayerNormalization(epsilon=1e-6)(x1)
        ffn = Dense(ff_dim, activation='relu')(x1)
        ffn = Dense(head_size * num_heads)(ffn)
        ffn = Dropout(0.1)(ffn)
        x = Add()([x1, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs, name='Transformer')

builders = {'1D-CNN': build_cnn, 'RNN': build_rnn, 'LSTM': build_lstm, 'Transformer': build_transformer}

model = builders[model_name](input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0)
]

history = model.fit(X_train, y_train, validation_split=0.15,
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   callbacks=callbacks, verbose=1, shuffle=False)

y_pred_scaled = model.predict(X_test, verbose=0).flatten()
y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

mae = mean_absolute_error(y_test_orig, y_pred_orig)
rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
mape = mean_absolute_percentage_error(y_test_orig, y_pred_orig) * 100

print(f"\n  Results: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

result = {
    'model': model_name, 'horizon': horizon_name,
    'MAE': float(mae), 'RMSE': float(rmse), 'MAPE': float(mape),
    'history': {'loss': [float(v) for v in history.history['loss']],
                'val_loss': [float(v) for v in history.history['val_loss']]},
    'y_test': y_test_orig.tolist(),
    'y_pred': y_pred_orig.tolist()
}

outpath = os.path.join(RESULTS_DIR, f'{model_name}_{horizon_name}.json')
with open(outpath, 'w') as f:
    json.dump(result, f)
print(f"  Saved: {outpath}")
