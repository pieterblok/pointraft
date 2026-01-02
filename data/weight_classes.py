import os
from pathlib import Path
import pandas as pd


def assign_weight_classes_to_splits(ground_truth_csv, target_col, last_bin_max, bin_distance=50):
    target_df = pd.read_csv(ground_truth_csv)
    max_weight = target_df[target_col].max()
    
    bins = list(range(0, last_bin_max + bin_distance, bin_distance))
    if bins[-1] < max_weight:
        bins.append(max_weight + 1)
    
    labels = range(len(bins) - 1)
    
    def get_weight_class(weight):
        for i in range(len(bins) - 1):
            if bins[i] <= weight < bins[i + 1]:
                return labels[i]
        return None
    
    target_df['weight_class'] = target_df[target_col].apply(get_weight_class)
    target_df.to_csv(ground_truth_csv, index=False)
    print(f"Updated ground truth CSV with weight classes saved to {ground_truth_csv}.")



if __name__ == "__main__":
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    datafolder = os.path.join(project_root, 'data', '3DPotatoTwin')

    ground_truth_csv = os.path.join(datafolder, 'ground_truth.csv')
    target_col = 'weight_g_inctack'  
    last_bin_max = 450

    assign_weight_classes_to_splits(ground_truth_csv, target_col, last_bin_max, bin_distance=50)