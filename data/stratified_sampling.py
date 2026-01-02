import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin')

    csv_file_path = os.path.join(datafolder, 'ground_truth.csv')
    df = pd.read_csv(csv_file_path)

    num_bins = 10
    weight_bins, bin_edges = pd.qcut(df["weight_g_inctack"], q=num_bins, labels=False, retbins=True, duplicates='drop')
    df['weight_bin'] = weight_bins
    counts = df['weight_bin'].value_counts().sort_index()


    ## Split the dataset into 60% training, 20% validation, and 20% test using stratified sampling based on the weight bins
    train, valtest = train_test_split(df, test_size=0.4, stratify=df["weight_bin"], random_state=20)
    test, val = train_test_split(valtest, test_size=0.5, stratify=valtest["weight_bin"], random_state=20)


    ## Visualize the splits
    plt.figure(figsize=(10, 6))
    sns.kdeplot(train["weight_g_inctack"], label="Training set", color='blue', fill=True, alpha=0.4)
    sns.kdeplot(val["weight_g_inctack"], label="Validation set", color='green', fill=True, alpha=0.4)
    sns.kdeplot(test["weight_g_inctack"], label="Test set", color='red', fill=True, alpha=0.4)

    plt.xlabel('Weight [g]')
    plt.ylabel('Density')
    plt.xlim(left=0, right=700) 
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    ## Write to csv
    train_split = pd.DataFrame({"label": train["label"], "split": "train"})
    val_split = pd.DataFrame({"label": val["label"], "split": "val"})
    test_split = pd.DataFrame({"label": test["label"], "split": "test"})
    splits_df = pd.concat([train_split, val_split, test_split], ignore_index=True)
    splits_df.to_csv(os.path.join(datafolder, 'splits.csv'), index=False)