import os
import gc
import time
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, optimizers, backend as K

NUM_CLASSES = 10
INPUT_LENGTH = 2048
TRAIN_SAMPLES = 8000 
VAL_SAMPLES = 2000
EPOCHS = 10
BATCH_SIZE = 32
OUTPUT_DIR = "research_v2_results_3_8000_2000"
os.makedirs(OUTPUT_DIR, exist_ok=True)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except:
            pass

class AGNBModel(models.Model):
    def __init__(self, inputs, outputs, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
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

def run_experiment():
    x_train, y_train = get_data(TRAIN_SAMPLES)
    x_val, y_val = get_data(VAL_SAMPLES)
    models_list = ["WDCNN", "TICNN", "DCN", "SRDCNN", "STIM", "STFT", "RNN_WDCNN"]
    optimizers_list = ["Adam", "SGD", "RMSprop"]
    
    perf_data = []
    
    for m_name in models_list:
        for opt_name in optimizers_list:
            tag = f"{m_name}_{opt_name}"
            start_t = time.time()
            print(f"Executing: {tag}")
            
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
            
            dur = time.time() - start_t
            best_acc = max(history.history['val_accuracy'])
            
            model.save_weights(os.path.join(OUTPUT_DIR, f"{tag}.weights.h5"))
            with open(os.path.join(OUTPUT_DIR, f"{tag}_log.json"), "w") as f:
                json.dump(history.history, f)
            
            perf_data.append({
                "model": m_name,
                "optimizer": opt_name,
                "accuracy": best_acc,
                "time": dur,
                "history": history.history
            })
            
            K.clear_session()
            del model
            gc.collect()
            
    return perf_data

def plot_visuals(data):
    models_list = sorted(list(set(d['model'] for d in data)))
    opts_list = sorted(list(set(d['optimizer'] for d in data)))
    
    plt.figure(figsize=(12, 6))
    for opt in opts_list:
        accs = [d['accuracy'] for d in data if d['optimizer'] == opt]
        plt.bar(np.arange(len(models_list)) + opts_list.index(opt)*0.2, accs, width=0.2, label=opt)
    plt.xticks(np.arange(len(models_list)) + 0.2, models_list, rotation=45)
    plt.title("Chart 1: Model Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_accuracy_comparison.png")

    plt.figure(figsize=(10, 6))
    avg_acc_model = [np.mean([d['accuracy'] for d in data if d['model'] == m]) for m in models_list]
    plt.bar(models_list, avg_acc_model, color='skyblue')
    plt.title("Chart 2: Average Accuracy per Model Architecture")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_avg_model_acc.png")

    plt.figure(figsize=(8, 6))
    avg_acc_opt = [np.mean([d['accuracy'] for d in data if d['optimizer'] == o]) for o in opts_list]
    plt.pie(avg_acc_opt, labels=opts_list, autopct='%1.1f%%', colors=['gold', 'lightgreen', 'coral'])
    plt.title("Chart 3: Optimizer Performance Contribution")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_optimizer_pie.png")

    plt.figure(figsize=(12, 6))
    times = [d['time'] for d in data]
    labels = [f"{d['model']}\n({d['optimizer']})" for d in data]
    plt.stackplot(range(len(times)), times, labels=['Training Duration'])
    plt.xticks(range(len(times)), labels, rotation=90, fontsize=8)
    plt.title("Chart 4: Training Time Profile (Seconds)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_training_time.png")

    plt.figure(figsize=(10, 8))
    pivot_table = np.zeros((len(models_list), len(opts_list)))
    for d in data:
        pivot_table[models_list.index(d['model']), opts_list.index(d['optimizer'])] = d['accuracy']
    sns.heatmap(pivot_table, annot=True, xticklabels=opts_list, yticklabels=models_list, cmap="YlGnBu")
    plt.title("Chart 5: Accuracy Heatmap (Model vs Optimizer)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_accuracy_heatmap.png")

if __name__ == "__main__":
    results = run_experiment()
    plot_visuals(results)
    print(f"Task completed. Files saved in {OUTPUT_DIR}")