import os
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, optimizers, backend as K
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 設定參數
NUM_CLASSES = 10
INPUT_LENGTH = 2048
TRAIN_SAMPLES = 500  
VAL_SAMPLES = 100

NUM_PARTICLES = 6
NUM_ITERATIONS = 5
EPOCHS = 5

# BOUNDS: [learning rate, dropout, batch size]
BOUNDS = np.array([
    [1e-4, 1e-2],
    [0.1, 0.5],
    [16, 64]
])
V_MAX = (BOUNDS[:, 1] - BOUNDS[:, 0]) * 0.2

# 產生合成資料
def get_data():
    def create_set(n):
        X, Y = [], []
        t = np.arange(INPUT_LENGTH)
        for c in range(NUM_CLASSES):
            f1, f2 = 0.01 * (c + 1), 0.05 * (c + 1)
            for _ in range(n):
                sig = (np.sin(2 * np.pi * f1 * t) + 
                       0.5 * np.cos(2 * np.pi * f2 * t) + 
                       0.3 * np.random.randn(INPUT_LENGTH))
                X.append(sig)
                Y.append(c)
        X = np.array(X, dtype=np.float32)[..., None]
        Y = np.array(Y, dtype=np.int32)
        return X, Y

    xt, yt = create_set(TRAIN_SAMPLES)
    xv, yv = create_set(VAL_SAMPLES)
    idx = np.random.permutation(len(xt))
    return (xt[idx], yt[idx]), (xv, yv)

(x_train, y_train), (x_val, y_val) = get_data()

# 建立一維卷積模型
def build_model(lr, dr):
    inputs = layers.Input(shape=(INPUT_LENGTH, 1))

    x = layers.Conv1D(16, 64, strides=16, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    
    res = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(res)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.Add()([res, x])
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.Dropout(dr)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 適應度函數：回傳驗證集最高準確率
def fitness(pos):
    lr, dr, bs = pos
    bs = int(np.clip(np.round(bs), 16, 64))

    model = build_model(lr, dr)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=bs,
        verbose=0
    )
    score = np.max(history.history['val_accuracy'])
    
    # 釋放記憶體避免 OOM
    K.clear_session()
    del model
    gc.collect()
    
    return score

# 粒子群最佳化演算法
def pso(mode="basic"):
    dim = len(BOUNDS)
    pos = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1], (NUM_PARTICLES, dim))
    vel = np.random.uniform(-V_MAX, V_MAX, (NUM_PARTICLES, dim))

    p_best = pos.copy()
    p_score = np.zeros(NUM_PARTICLES)
    g_best = None
    g_score = -1
    history = []

    # 初始化
    for i in range(NUM_PARTICLES):
        p_score[i] = fitness(pos[i])
        if p_score[i] > g_score:
            g_score = p_score[i]
            g_best = pos[i].copy()
    history.append(g_score)

    # 迭代更新
    for t in range(NUM_ITERATIONS):
        w = 0.9 - 0.5 * (t / max(NUM_ITERATIONS - 1, 1))

        for i in range(NUM_PARTICLES):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = (w * vel[i] + 
                      1.5 * r1 * (p_best[i] - pos[i]) + 
                      1.5 * r2 * (g_best - pos[i]))
            
            vel[i] = np.clip(vel[i], -V_MAX, V_MAX)
            pos[i] = np.clip(pos[i] + vel[i], BOUNDS[:, 0], BOUNDS[:, 1])

            score = fitness(pos[i])
            if score > p_score[i]:
                p_score[i] = score
                p_best[i] = pos[i].copy()
            if score > g_score:
                g_score = score
                g_best = pos[i].copy()

        history.append(g_score)
        print(f"[{mode}] iter {t+1}: {g_score:.4f}")

    return g_best, g_score, history

# 執行流程
print("Running baseline...")
baseline_model = build_model(1e-3, 0.3)
baseline_model.fit(x_train, y_train, epochs=EPOCHS, verbose=0, validation_data=(x_val, y_val))
b_acc = max(baseline_model.history.history['val_accuracy'])
baseline_model.save("baseline.keras")  # 更新為較新的 Keras 存檔格式

print("Running PSO...")
best_pos, best_score, hist = pso("pso")

print("Training Final Model...")
final_model = build_model(best_pos[0], best_pos[1])
final_model.fit(x_train, y_train, epochs=EPOCHS, verbose=0, validation_data=(x_val, y_val))
final_model.save("pso_model.keras")

# 繪製收斂圖
plt.plot(hist)
plt.title("PSO Convergence")
plt.xlabel("Iteration")
plt.ylabel("Val Accuracy")
plt.savefig("convergence.png")
plt.show()

# 繪製混淆矩陣
pred = np.argmax(final_model.predict(x_val), axis=1)
cm = confusion_matrix(y_val, pred)
ConfusionMatrixDisplay(cm).plot()
plt.savefig("cm.png")
plt.show()

# 輸出紀錄
with open("log.txt", "w") as f:
    f.write(f"Baseline: {b_acc}\n")
    f.write(f"PSO: {best_score}\n")
    f.write(f"Best Params (lr, dr, bs): {best_pos}\n")