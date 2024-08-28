import argparse
import torch
import os
import numpy as np
import pandas as pd
from utils.utils import printlog, load_data, import_model, init_dl_program
from experiments.utils_downstream import eval_classification, eval_cluster, plot_confusion_matrix, plot_tsne

from experiments.configs.ts2vec_expconfigs import allts2vec_expconfigs
from experiments.configs.tnc_expconfigs import alltnc_expconfigs
from experiments.configs.cpc_expconfigs import allcpc_expconfigs
from experiments.configs.simclr_expconfigs import allsimclr_expconfigs
from experiments.configs.slidingmse_expconfigs import allslidingmse_expconfigs
from experiments.configs.rebar_expconfigs import allrebar_expconfigs
from sklearn.manifold import TSNE
import time

all_expconfigs = {**allts2vec_expconfigs, **alltnc_expconfigs, **allcpc_expconfigs, **allsimclr_expconfigs, **allslidingmse_expconfigs, **allrebar_expconfigs}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str, required=True)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    args = parser.parse_args()

    # selecting config according to arg
    config = all_expconfigs[args.config]
    config.set_rundir(args.config)

    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # Begin training contrastive learner
    if (args.retrain == True) or (not os.path.exists(os.path.join("experiments/out/", config.data_name, config.run_dir, "checkpoint_best.pkl"))):
        train_data, _, val_data, _, _, _ = load_data(config=config, data_type="fullts")
        model = import_model(config, train_data=train_data, val_data=val_data)
        start = time.time()
        model.fit()
        end = time.time()
        log_dir = model.run_dir
        training_time = round(end - start,3)
        printlog(f"time to fit the model: {training_time} seconds", path=log_dir)

    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data(config=config, data_type="subseq")
    model = import_model(config, reload_ckpt=True)

    run_dir = model.run_dir

    # Construct variant string
    method_neighbours = 'adf' if config.adf else 'sim'
    variant = f"{config.model_type}-{config.encoder_type}-{method_neighbours} {config.w}"

    # Evaluate classification
    eval_class_dict = eval_classification(model=model, savepath=model.run_dir,
                                          train_data=train_data, train_labels=train_labels,
                                          val_data=val_data, val_labels=val_labels,
                                          test_data=test_data, test_labels=test_labels,
                                          reencode=args.retrain)

    # Add variant to the evaluation dictionary
    eval_class_dict['variant'] = variant
    eval_class_dict['training_time'] = training_time

    # Log classification results
    printlog("-------------------------------", path=run_dir)
    printlog(f"Classification Results with Linear Probe", path=run_dir)
    printlog('Accuracy: ' + str(eval_class_dict['acc']), path=run_dir)
    printlog('AUROC: ' + str(eval_class_dict['auroc']), path=run_dir)
    printlog('AUPRC: ' + str(eval_class_dict['auprc']), path=run_dir)
    printlog('Balanced Accuracy: ' + str(eval_class_dict['balanced_acc']), path=run_dir)
    printlog('F1 Score: ' + str(eval_class_dict['f1']), path=run_dir)
    printlog('Variant: ' + variant, path=run_dir)
    printlog('Training time: ' + str(training_time), path=run_dir)
    printlog("-------------------------------", path=run_dir)

    # Save classification metrics to CSV
    metrics_file = os.path.join(run_dir, 'classification_metrics.csv')
    pd.DataFrame([eval_class_dict]).to_csv(metrics_file, index=False)

    # Plot t-SNE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_title = f"{config.model_type}-{config.encoder_type}-{method_neighbours} with W= {config.w}"
    # plot_path = os.path.join(run_dir, f"{config.data_name}_{config.model_type}_{config.encoder_type}_{method_neighbours}_{str(config.w)}_tsne")
    plot_path = run_dir
    plot_tsne(model=model, x_data=test_data, y_data=test_labels, savepath=plot_path, window_size=config.subseq_size, device=device, title=plot_title)

