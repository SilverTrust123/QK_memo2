import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, backend as K

# GPU 硬體配置
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus
    try
        for gpu in gpus
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e
        print(e)

# 實驗參數
NUM_CLASSES = 10
INPUT_LENGTH = 2048
TRAIN_SAMPLES = 700
VAL_SAMPLES = 200
NUM_PARTICLES = 10
NUM_ITERATIONS = 10
EPOCHS = 10
BOUNDS = np.array([[1e-4, 1e-2], [0.1, 0.5], [16, 128]])
V_MAX = (BOUNDS[, 1] - BOUNDS[, 0])  0.2

# 資料生成
def get_data()
    def create_set(n)
        X, Y = [], []
        t = np.arange(INPUT_LENGTH)
        for c in range(NUM_CLASSES)
            f1, f2 = 0.01  (c + 1), 0.05  (c + 1)
            for _ in range(n)
                sig = np.sin(2np.pif1t) + 0.5np.cos(2np.pif2t) + 0.3np.random.randn(INPUT_LENGTH)
                X.append(sig)
                Y.append(c)
        return np.expand_dims(np.array(X, dtype='float32'), -1), np.array(Y, dtype='int32')
    (xt, yt) = create_set(TRAIN_SAMPLES)
    (xv, yv) = create_set(VAL_SAMPLES)
    idx = np.arange(len(xt))
    np.random.shuffle(idx)
    return (xt[idx], yt[idx]), (xv, yv)

(x_train, y_train), (x_val, y_val) = get_data()

# 模型構建
def build_model(lr, dr)
    inputs = layers.Input(shape=(INPUT_LENGTH, 1))
    x = layers.Conv1D(16, 64, strides=16, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x_res = layers.MaxPooling1D(2)(x)
    c5 = layers.Conv1D(64, 3, padding='same', activation='relu')(x_res)
    c6 = layers.Conv1D(64, 3, padding='same', activation='relu')(c5)
    x = layers.Add()([x_res, c6])
    x = layers.Activation('relu')(x)
    x = layers.Conv1D(200, 1, activation='relu')(x)
    x = layers.Dropout(dr)(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def fitness(pos)
    m = build_model(pos[0], pos[1])
    h = m.fit(x_train, y_train, batch_size=int(np.round(pos[2])), epochs=EPOCHS, validation_data=(x_val, y_val), verbose=0)
    acc = h.history[val_accuracy][-1]
    K.clear_session()
    del m
    gc.collect()
    return acc

# PSO 引擎
def pso_engine(mode='original')
    dim = len(BOUNDS)
    pos = np.random.uniform(BOUNDS[, 0], BOUNDS[, 1], (NUM_PARTICLES, dim))
    vel = np.random.uniform(-V_MAX, V_MAX, (NUM_PARTICLES, dim))
    p_pos, p_val = np.copy(pos), np.zeros(NUM_PARTICLES)
    g_pos, g_val = np.zeros(dim), -1.0
    hist = []
    
    for i in range(NUM_PARTICLES)
        p_val[i] = fitness(pos[i])
        if p_val[i]  g_val
            g_val, g_pos = p_val[i], np.copy(pos[i])
    hist.append(g_val)

    for t in range(NUM_ITERATIONS)
        w = (0.9 - t  (0.5  (NUM_ITERATIONS - 1))) if mode == 'improved_ldw' else 0.8
        for i in range(NUM_PARTICLES)
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[i] = w  vel[i] + 1.5  r1  (p_pos[i] - pos[i]) + 1.5  r2  (g_pos - pos[i])
            vel[i] = np.clip(vel[i], -V_MAX, V_MAX)
            pos[i] = np.clip(pos[i] + vel[i], BOUNDS[, 0], BOUNDS[, 1])
            acc = fitness(pos[i])
            if acc  p_val[i]
                p_val[i], p_pos[i] = acc, np.copy(pos[i])
            if acc  g_val
                g_val, g_pos = acc, np.copy(pos[i])
        hist.append(g_val)
        print(f{mode} Iter {t+1} {g_val.4f})
    return g_pos, g_val, hist

# 執行實驗
# 1. Baseline
print(Running Baseline...)
b_acc = fitness([0.001, 0.3, 64])
m_b = build_model(0.001, 0.3)
m_b.save('model_baseline.h5')

# 2. Original PSO
print(Running Original PSO...)
p_o_pos, p_o_val, h_o = pso_engine('original')
m_o = build_model(p_o_pos[0], p_o_pos[1])
m_o.save('model_pso_original.h5')

# 3. Improved PSO
print(Running Improved PSO...)
p_i_pos, p_i_val, h_i = pso_engine('improved_ldw')
m_i = build_model(p_i_pos[0], p_i_pos[1])
m_i.save('model_pso_improved.h5')

# 繪圖與存檔
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(h_o, label='Original')
plt.plot(h_i, label='Improved (LDW)')
plt.title('Convergence')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Base', 'Orig', 'Improv'], [b_acc, p_o_val, p_i_val])
plt.title('Accuracy Comparison')
plt.savefig('experiment_results.png')
plt.show()

# 存儲實驗數據備查
with open('log.txt', 'w') as f
    f.write(fBaseline {b_acc}nOriginal PSO {p_o_val} Params {p_o_pos}nImproved PSO {p_i_val} Params {p_i_pos})