import os
import torch
import numpy as np
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score, average_precision_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from utils.utils import printlog
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
def plot_confusion_matrix(true_labels, predicted_labels, classes, savepath="", normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap)
    plt.title(title)
    plt.savefig(os.path.join(savepath, 'confusion_matrix.png')) if savepath else plt.show()


def eval_classification(model, train_data, train_labels, val_data, val_labels, test_data, test_labels, 
                        savepath="", reencode=True):

    if not reencode and os.path.exists(os.path.join(savepath, "encoded_train.pth")):
        test_repr = torch.load(os.path.join(savepath, "encoded_test.pth"))
        train_repr = torch.load(os.path.join(savepath, "encoded_train.pth"))
        val_repr = torch.load(os.path.join(savepath, "encoded_val.pth"))
    else:
        train_repr = model.encode(train_data)
        val_repr = model.encode(val_data)
        test_repr = model.encode(test_data)
        printlog("Encoding train, val, test ...", savepath)
        torch.save(train_repr, os.path.join(savepath, "encoded_train.pth"), pickle_protocol=4)
        torch.save(val_repr, os.path.join(savepath, "encoded_val.pth"), pickle_protocol=4)
        torch.save(test_repr, os.path.join(savepath, "encoded_test.pth"), pickle_protocol=4)


    linearprobe_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=10,
            max_iter=1000,
            multi_class='multinomial',
            class_weight="balanced",
            verbose=0,
        )
    )

    printlog("Training linear probe ... ", savepath)
    linearprobe_classifier.fit(train_repr, train_labels)
    y_pred = linearprobe_classifier.predict(test_repr)
    y_score = linearprobe_classifier.predict_proba(test_repr)
    acc = linearprobe_classifier.score(test_repr, test_labels)

    test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    if train_labels.max()+1 == 2:
        test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+2))
        test_labels_onehot = test_labels_onehot[:, :2]
    auprc = average_precision_score(test_labels_onehot, y_score)
    auroc = roc_auc_score(test_labels_onehot, y_score)
    
    balanced_acc = balanced_accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average='weighted')

    return {'acc': acc, 'auprc': auprc, "auroc": auroc, 'balanced_acc': balanced_acc, 'f1': f1}

def eval_cluster(model, train_data, train_labels, 
                        val_data, val_labels, 
                        test_data, test_labels, 
                        savepath="", k=None, reencode=False):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    

    if not reencode and os.path.exists(os.path.join(savepath, "encoded_train.pth")):
        test_repr = torch.load(os.path.join(savepath, "encoded_test.pth"))
        train_repr = torch.load(os.path.join(savepath, "encoded_train.pth"))
        val_repr = torch.load(os.path.join(savepath, "encoded_val.pth"))
    else:
        train_repr = model.encode(train_data)
        val_repr = model.encode(val_data)
        test_repr = model.encode(test_data)
        printlog("Encoding train, val, test ...", savepath)
        torch.save(train_repr, os.path.join(savepath, "encoded_train.pth"), pickle_protocol=4)
        torch.save(val_repr, os.path.join(savepath, "encoded_val.pth"), pickle_protocol=4)
        torch.save(test_repr, os.path.join(savepath, "encoded_test.pth"), pickle_protocol=4)


    if k == None:
        k = len(np.unique(test_labels))

    # printlog("Running k-means algorithm ... ", savepath)
    kmeans = KMeans(n_clusters=k, random_state=10, n_init="auto").fit(test_repr) # (710, 320) test_repr shape
    cluster_labels = kmeans.labels_
    s_score = silhouette_score(test_repr, cluster_labels)
    db_score = davies_bouldin_score(test_repr, cluster_labels)
    ar_score = adjusted_rand_score(cluster_labels, test_labels)
    nmi_score = normalized_mutual_info_score(cluster_labels, test_labels)

    return {"sil": s_score,  "db": db_score,  "ari": ar_score,  "nmi": nmi_score, "k":k}

def plot_tsne(model, x_data, y_data, savepath="", window_size=None, device=None, augment=4, cv=0,title=''):

    # Encode windows using the model
    encodings = model.encode(x_data)

    # Apply t-SNE to the encoded representations
    tsne = TSNE(n_components=2)
    embedding = tsne.fit_transform(encodings)
    original_embedding = TSNE(n_components=2).fit_transform(x_data.reshape(len(x_data),-1))
    state_labels = {
        0: "Walking",
        1: "Walking Upstairs",
        2: "Walking Donwstairs",
        3: "Sitting",
        4: "Standing",
        5: "Laying"
    }
    df_original = pd.DataFrame({"f1": original_embedding[:, 0], "f2": original_embedding[:, 1], "state": y_data})
    df_original['state'] = df_original['state'].map(state_labels)

    # Prepare data for plotting
    df_encoding = pd.DataFrame({"f1": embedding[:, 0], "f2": embedding[:, 1], "state": y_data})  # Assuming y_data contains labels for each sample
    df_encoding['state'] = df_encoding['state'].map(state_labels)
    # Save plots
    plot_dir = os.path.join(savepath)
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots()
    # "TNC-TS2Vec-sim" "TNC-TS2Vec-adf" "TNC-RNN-adf" "TNC-RNN-sim"
    ax.set_title(title, fontweight="bold", fontsize=18)
    sns.scatterplot(x="f1", y="f2", data=df_encoding, hue="state", palette='Set2', ax=ax,s=30, edgecolor='none')
    ax.legend([],[], frameon=False) 
    plt.savefig(os.path.join(plot_dir, "encoding_distribution.pdf"), bbox_inches='tight')
