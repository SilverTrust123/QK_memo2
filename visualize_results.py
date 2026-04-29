import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_all(log_root):
    data = []
    files = glob.glob(f"{log_root}/**/test_result/log/version_0/metrics.csv", recursive=True)
    for f in files:
        p = f.split(os.sep)
        df = pd.read_csv(f)
        if 'test_acc' in df.columns:
            data.append({"Dataset": p[-8], "Model": p[-7], "Size": int(p[-6].replace("size_","")), "Optimizer": p[-5], "Accuracy": df['test_acc'].iloc[-1]})
    
    res_df = pd.DataFrame(data)

    # 圖 A: 模型與優化器對比 (論文 Fig 6 類型)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=res_df, x="Model", y="Accuracy", hue="Optimizer")
    plt.title("Optimizer Accuracy Distribution per Model")
    plt.savefig("paper_fig_optimizer_dist.png")

    # 圖 B: 樣本規模對性能的影響 (論文 Fig 4 類型)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=res_df, x="Size", y="Accuracy", hue="Optimizer", marker='o')
    plt.title("Impact of Training Sample Size")
    plt.savefig("paper_fig_sample_size.png")

    # 圖 C: 訓練收斂曲線 (從第一個實驗中提取)
    # 此處需額外讀取 training 下的 metrics.csv
    first_log = glob.glob(f"{log_root}/**/training/log/version_0/metrics.csv", recursive=True)[0]
    train_df = pd.read_csv(first_log)
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['epoch'], train_df['train_loss'])
    plt.title("Example Training Convergence (Loss)")
    plt.savefig("paper_fig_convergence.png")

if __name__ == "__main__":
    plot_all("./EXPERIMENT_RESULTS")