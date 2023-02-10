import torch
import numpy as np
import matplotlib.pyplot as plt


def show_histogram(output_file, list_data, epoch, mode):

    with open(list_data, "r") as f:
        tamp_list = [t.strip().split(',') for t in f.readlines()]

    list_labels, list_scores = [], []
    for index in range(0, len(tamp_list)):
        if mode == "cls":
            list_labels.append(int(tamp_list[index][1]))
            list_scores.append(float(tamp_list[index][2]))
        elif mode == "combined":
            list_labels.append(int(tamp_list[index][2]))
            list_scores.append(float(tamp_list[index][3]))

    negative_pred, positive_pred = [], []
    for i, t in enumerate(list_scores):
        if list_labels[i] == 0:
            negative_pred.append(t)
        else:
            positive_pred.append(t)

    plt.figure(figsize=(10,10))
    plt.hist(np.array(negative_pred), bins=100, density=False, alpha=0.5, label='Single')
    plt.hist(np.array(positive_pred), bins=100, density=False, alpha=0.5,label='Double')
    plt.legend(['Single', 
                'Double'])
    
    if epoch is None:
        plt.savefig(str(output_file)+'/histogram.png')
    else:
        plt.savefig(str(output_file)+"/histogram"+epoch+".png")

    


if __name__ == '__main__':
    show_histogram()