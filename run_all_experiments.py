import os
import shutil
import subprocess
import pandas as pd
import time
from experiments.configs.tnc_expconfigs import alltnc_expconfigs

# Define the different combinations of parameters
w_values = [0.2, 0.05]
encoder_types = ['RNN', 'TS2Vec']
adf_values = [True, False]
num_runs = 8  # Number of times to run each variant

# output file
csv_filename = "experiments/out/experiment_results.csv"

results_df = pd.DataFrame(columns=["variant", "acc", "auprc", "auroc", "balanced_acc", "f1", "training_time"])
results_df.to_csv(csv_filename, index=False)

# Iterate over all combinations of w, encoder_type, and adf
for w in w_values:
    for encoder_type in encoder_types:
        for adf in adf_values:
            variant_name = f"TNC-{encoder_type}-{'adf' if adf else 'sim'}-{w}"

            # Configure the key for this variant to be found at alltnc_expconfigs
            config_key = f"tnc_har_{encoder_type}_{'adf' if adf else 'sim'}_{w}"
            config = alltnc_expconfigs[config_key]

            # Run the experiment multiple times
            for run in range(num_runs):
                print(f"Running experiment {run+1} for variant {variant_name}...")
                
                
                # Use subprocess to run the experiment
                subprocess.run(["python", "run_exp.py", "-c", config_key, "--retrain"])
                
                output_dir = f"experiments/out/har/{config_key}"
                metrics_path = os.path.join(output_dir, "classification_metrics.csv")
                run_dir = os.path.join(output_dir, f"run_{run}")
                os.makedirs(run_dir, exist_ok=True)

                if os.path.exists(metrics_path):
                    metrics_df = pd.read_csv(metrics_path)
                    # Move all files from the output_dir to run_dir
                    for file_name in os.listdir(output_dir):
                        file_path = os.path.join(output_dir, file_name)
                        if os.path.isfile(file_path):
                            shutil.move(file_path, os.path.join(run_dir, file_name))
                    
                    if all(metric in metrics_df.columns for metric in ["acc", "auprc", "auroc", "balanced_acc", "f1","training_time"]):
                        metrics = metrics_df.iloc[0]
                        acc = metrics["acc"]
                        auprc = metrics["auprc"]
                        auroc = metrics["auroc"]
                        balanced_acc = metrics["balanced_acc"]
                        f1 = metrics["f1"]
                        training_time = metrics["training_time"]

                        new_row = pd.DataFrame([[variant_name, acc, auprc, auroc, balanced_acc, f1, training_time]],
                                               columns=["variant", "acc", "auprc", "auroc", "balanced_acc", "f1", "training_time"])
                        results_df = pd.concat([results_df, new_row], ignore_index=True)
                   
                        # Save the current updated results to the CSV file
                        results_df.to_csv(f'{output_dir}/{csv_filename[:-4]}_{run}.csv', index=False)
                    else:
                        print(f"Metrics file {metrics_path} is missing some required columns.")
                else:
                    print(f"Metrics file {metrics_path} does not exist.")

            print(f"Completed {num_runs} runs for variant {variant_name}")

print(f"All experiments completed and saved to {csv_filename}.")
results_df.to_csv(f'{csv_filename}', index=False)
