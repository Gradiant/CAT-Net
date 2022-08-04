import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd


list_single, list_double = [], []

# img_name: image name, weight, heigth, qf
with open("/media/data/workspace/rroman/CAT-Net/data_DOCIMANv1.txt", "r") as f:
    tamp_list = [t.strip().split(',') for t in f.readlines()]

classification = True
list_single, list_double = [], []
if classification:
    for index in range(0,len(tamp_list)):
        label = int(tamp_list[index][1])
        pred = 1 if float(tamp_list[index][2]) >= 0.5 else 0
        if label == 1:
            qf1 = tamp_list[index][0].split('_')[-3]
            qf2 = tamp_list[index][0].split("_")[-1].split(".")[0]
            list_double.append((qf1, qf2, label, pred))
        if label == 0:
            qf1 = tamp_list[index][0].split('_')[-1].split(".")[0]
            list_single.append((qf1, label, pred))

    
    df= pd.DataFrame(list_double, columns=['qf1', 'qf2', 'label', 'pred']) # creamos dataframe con los datos de cada imagen de evaluación
    df_count = df.groupby(['qf1', 'qf2'], as_index=False)[["label"]].count() # saber cuántas imágenes hay para cada combinación de qf
    df['correct'] = df['label'] * df['pred'] # nuevo campo con valor 0/1
    df_correct = df.groupby(['qf1', 'qf2'], as_index=False)[["correct"]].sum() # suma de True Positives por qfs
    df_concated = pd.concat([df_count, df_correct['correct']], axis=1) 
    df_concated['accuracy'] = df_concated['correct'] / df_concated['label']
    print(df_concated)
    df_concated.to_excel("/media/data/workspace/rroman/CAT-Net/qf_results_DOCIMANv1.xlsx")  

    df_single= pd.DataFrame(list_single, columns=['qf1', 'label', 'pred']) # creamos dataframe con los datos de cada imagen de evaluación
    df_count = df_single.groupby(['qf1'], as_index=False)[["label"]].count() # saber cuántas imágenes hay para cada qf1
    df_single.loc[df_single['label'] == df_single['pred'], 'correct'] = 1 
    df_single.loc[df_single['label'] != df_single['pred'], 'correct'] = 0
    df_correct = df_single.groupby(['qf1'], as_index=False)[["correct"]].sum() # suma de TP por qfs
    df_concated = pd.concat([df_count, df_correct['correct']], axis=1) 
    df_concated['accuracy'] = df_concated['correct'] / df_concated['label']
    print(df_concated)
    df_concated.to_excel("/media/data/workspace/rroman/CAT-Net/qf_results_DOCIMANv1_single.xlsx")  
else:
    df= pd.DataFrame(tamp_list, columns=['img_name', 'qf1', 'qf2', 'mean_IoU', 'p_mIoU', 'p_AP', 'p_f1'], dtype=float)
    df_mean = df.groupby(['qf1', 'qf2'], as_index=False)[["mean_IoU", "p_mIoU", "p_AP", "p_f1"]].mean()
    df_mean.to_excel("/media/data/workspace/rroman/CAT-Net/qf_results_DOCUMENTS_cm.xlsx")








    
