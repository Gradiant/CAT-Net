import numpy as np
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt



with open("/media/data/workspace/rroman/CAT-Net/data_DOCUMENTS_cls.txt", "r") as f:
    tamp_list = [t.strip().split(',') for t in f.readlines()]
    
    
list_img_names, list_scores, list_labels = [], [], []
for index in range(0, len(tamp_list)):
    list_labels.append(int(tamp_list[index][1]))
    list_scores.append(float(tamp_list[index][2]))
    # if int(tamp_list[index][1]) == 1:
    #     img_names = tamp_list[index][0]
    #     qf1 = tamp_list[index][0].split("_")[-3]
    #     print(qf1)
    #     qf2 = tamp_list[index][0].split("_")[-1].split(".")[0]
    #     if int(qf2) > 70 and int(qf1) > 70:
    #         list_labels.append(int(tamp_list[index][1]))
    #         list_scores.append(float(tamp_list[index][2]))

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
plt.title("DOCIMANv1 ROC curve QF1>70, QF2>70")
plt.legend(loc="lower right")
plt.savefig("/media/data/workspace/rroman/CAT-Net/roc_curve_DOCIMANv1_qf1_qf2_70.png")
