import os
import gc
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, optimizers, backend as K

NUM_CLASSES = 10
INPUT_LENGTH = 2048
TRAIN_SAMPLES = 1000 
VAL_SAMPLES = 200
EPOCHS = 10
BATCH_SIZE = 32
OUTPUT_DIR = "research_v2_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 強制 GPU 顯存自適應
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

class AGNBModel(models.Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.gradient_mu = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.rho = 0.99

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        gradients = tape.gradient(loss, self.trainable_variables)
        
        global_norm = tf.linalg.global_norm(gradients)
        self.gradient_mu.assign(self.rho * self.gradient_mu + (1 - self.rho) * global_norm)

        scale = self.gradient_mu / (global_norm + 1e-6)
        scale = tf.clip_by_value(scale, 0.5, 2.0)
        
        processed_gradients = [g * scale for g in gradients]

        self.optimizer.apply_gradients(zip(processed_gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

def build_structure(name):
    inputs = layers.Input(shape=(INPUT_LENGTH, 1))
    
    if name == "WDCNN":
        x = layers.Conv1D(16, 64, strides=16, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif name == "TICNN":
        x = layers.Conv1D(32, 64, padding='same', activation='relu')(inputs)
        x = layers.Dropout(0.3)(x)
        x = layers.GlobalAveragePooling1D()(x)
    elif name == "DCN":
        x = layers.Conv1D(32, 3, dilation_rate=2, padding='same', activation='relu')(inputs)
        x = layers.GlobalAveragePooling1D()(x)
    elif name == "SRDCNN":
        x1 = layers.Conv1D(16, 3, padding='same', activation='relu')(inputs)
        x2 = layers.Conv1D(16, 3, padding='same', activation='relu')(x1)
        x = layers.Add()([x1, x2])
        x = layers.GlobalAveragePooling1D()(x)
    elif name == "STIM":
        x = layers.Reshape((64, 32, 1))(inputs)
        x = layers.Conv2D(16, 3, activation='relu')(x)
        x = layers.Flatten()(x)
    elif name == "STFT":
        x = layers.Reshape((32, 64, 1))(inputs)
        x = layers.Conv2D(16, 3, activation='relu')(x)
        x = layers.Flatten()(x)
    elif name == "RNN_WDCNN":
        x = layers.AveragePooling1D(8)(inputs)
        x = layers.LSTM(32)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return AGNBModel(inputs, outputs)

def get_data(n_samples):
    x = np.random.normal(0, 1, (n_samples * NUM_CLASSES, INPUT_LENGTH, 1)).astype(np.float32)
    for i in range(NUM_CLASSES):
        freq = 0.05 * (i + 1)
        t = np.arange(INPUT_LENGTH)
        x[i*n_samples : (i+1)*n_samples, :, 0] += np.sin(2 * np.pi * freq * t)
    
    y = np.repeat(np.arange(NUM_CLASSES), n_samples).astype(np.int32)
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]

def execute_benchmark():
    print("正在加載故障數據 ")
    x_train, y_train = get_data(TRAIN_SAMPLES)
    x_val, y_val = get_data(VAL_SAMPLES)
    
    models_list = ["WDCNN", "TICNN", "DCN", "SRDCNN", "STIM", "STFT", "RNN_WDCNN"]
    optimizers_list = ["SGD", "Adam", "RMSProp"]
    
    results_matrix = {}

    for m_name in models_list:
        results_matrix[m_name] = {}
        for opt_name in optimizers_list:
            start_time = time.time()
            print(f"跑: {m_name} + {opt_name}...", end=" ", flush=True)
            
            model = build_structure(m_name)
            opt = getattr(optimizers, opt_name)(learning_rate=0.001)
            model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            history = model.fit(
                x_train, y_train, 
                validation_data=(x_val, y_val),
                epochs=EPOCHS, 
                batch_size=BATCH_SIZE, 
                verbose=0
            )
            
            best_val_acc = max(history.history['val_accuracy'])
            results_matrix[m_name][opt_name] = best_val_acc

            K.clear_session()
            del model
            gc.collect()
            print(f"完成! (準確率: {best_val_acc:.4f}, 耗時: {time.time()-start_time:.1f}s)")

    return results_matrix

def generate_report(results):
    print("\n" + "="*50)
    print(f"{'Model':<12} | {'SGD':<8} | {'Adam':<8} | {'RMSProp':<8}")
    print("-" * 50)
    for m, opts in results.items():
        print(f"{m:<12} | {opts['SGD']:.4f} | {opts['Adam']:.4f} | {opts['RMSProp']:.4f}")
    print("="*50)

    plt.figure(figsize=(12, 7))
    x_axis = np.arange(len(results))
    width = 0.25
    
    for i, opt in enumerate(["SGD", "Adam", "RMSProp"]):
        vals = [results[m][opt] for m in results]
        plt.bar(x_axis + i*width, vals, width, label=opt)
    
    plt.xticks(x_axis + width, list(results.keys()), rotation=30)
    plt.title("Benchmarking Mechanical Fault Diagnosis Models with AGNB Logic")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_benchmark.png")
    print(f"\n報告圖表已保存至: {OUTPUT_DIR}/final_benchmark.png")

if __name__ == "__main__":
    final_data = execute_benchmark()
    generate_report(final_data)