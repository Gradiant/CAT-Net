import numpy as np
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt



def plot_roc_curve(output_file, list_data, epoch):
    with open(list_data, "r") as f:
        tamp_list = [t.strip().split(',') for t in f.readlines()]

    list_scores, list_labels = [], []
    for index in range(0, len(tamp_list)):
        list_labels.append(int(tamp_list[index][1]))
        list_scores.append(float(tamp_list[index][2]))

    y = np.array(list_labels)
    scores = np.array(list_scores)
    fpr, tpr, thresholds = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve, thr=0.5")
    plt.legend(loc="lower right")
    if epoch == None:
        plt.savefig(str(output_file)+"/roc_curve.png")
    else:
        plt.savefig(str(output_file)+"/roc_curve_"+epoch+".png")

if __name__ == '__main__':
    plot_roc_curve()