import os
import torch
import numpy as np
import pandas as pd
import fdob
import info
import benchmark
from torchvision import transforms
import fdob.processing as processing

DATASETS = ["cwru", "mfpt"]
MODELS = list(info.model.keys())
OPTIMIZERS = list(info.hparam.keys())
DATA_SIZES = [16, 64, 128]
N_TRIALS = 1
N_EPOCHS = 50
BATCH_SIZE = 64
N_GPU = 0
LOG_ROOT = "./EXPERIMENT_RESULTS"

def get_balanced_subset(X, y, n_per_class):
    unique_labels = np.unique(y)
    indices = []
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        n = min(len(label_indices), n_per_class)
        selected = np.random.choice(label_indices, n, replace=False)
        indices.extend(selected)
    return X[indices], y[indices]

def main():
    data_raw = {
        "cwru": fdob.download_cwru("./data/cwru"),
        "mfpt": fdob.download_mfpt("./data/mfpt")
    }

    for ds_name in DATASETS:
        df = data_raw[ds_name]
        if ds_name == "cwru":
            df = df[(df["label"] != 999) & (df["load"] != 0)]
        
        train_df, val_df, test_df = fdob.split_dataframe(df, 0.6, 0.2)

        for m_name in MODELS:
            m_cfg = info.model[m_name]
            sl = m_cfg["sample_length"]
            
            X_tr_all, y_tr_all = fdob.build_from_dataframe(train_df, sl, sl//2, False)
            X_val, y_val = fdob.build_from_dataframe(val_df, sl, sl//2, False)
            X_te, y_te = fdob.build_from_dataframe(test_df, sl, sl//2, False)

            for d_size in DATA_SIZES:
                X_tr, y_tr = get_balanced_subset(X_tr_all, y_tr_all, d_size)
                
                dmodule = fdob.DatasetHandler()
                dmodule.assign(X_tr, y_tr, X_val, y_val, X_te, y_te, sl, 
                               ds_name, transforms.Compose(m_cfg["tf"]), 
                               transforms.Compose([processing.NpToTensor()]), BATCH_SIZE, 4)

                for opt_key in OPTIMIZERS:
                    h_cfg = info.hparam[opt_key]
                    sampled = fdob.log_qsample(h_cfg["n_params"], h_cfg["param_names"],
                                               h_cfg["lb"], h_cfg["ub"], h_cfg["reversed"], N_TRIALS)

                    for i in range(N_TRIALS):
                        save_dir = os.path.join(LOG_ROOT, ds_name, m_name, f"size_{d_size}", opt_key, f"trial_{i}")
                        
                        kwargs = {}
                        for p in h_cfg["param_names"]:
                            if opt_key == "adam" and p in ["beta1", "beta2"]:
                                kwargs["betas"] = (sampled["beta1"][i], sampled["beta2"][i])
                            else:
                                kwargs[p] = sampled[p][i]

                        benchmark.train(
                            dmodule.dataloaders[ds_name]["train"],
                            dmodule.dataloaders[ds_name]["val"],
                            m_cfg["model"], {"n_classes": 10},
                            h_cfg["optimizer"], kwargs,
                            torch.nn.CrossEntropyLoss, None,
                            N_EPOCHS, 6464, N_GPU, save_dir
                        )
                        
                        benchmark.test(dmodule.dataloaders[ds_name]["test"],
                                       m_cfg["model"], {"n_classes": 10},
                                       h_cfg["optimizer"], kwargs,
                                       torch.nn.CrossEntropyLoss, None, N_EPOCHS, 6464, N_GPU, save_dir, "test_result")

if __name__ == "__main__":
    main()