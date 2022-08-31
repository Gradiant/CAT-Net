import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def qf_analysis(output_folder, list_data, cls_mode=True, epoch=None):
    with open(list_data, "r") as f:
        tamp_list = [t.strip().split(',') for t in f.readlines()]

    if cls_mode:
        list_single, list_double = [], []
        for index in range(0,len(tamp_list)):
            label = int(tamp_list[index][1])
            pred = 1 if float(tamp_list[index][2]) >= 0.5 else 0
            if label == 1:
                qf1 = int(tamp_list[index][0].split('_')[-3])
                qf2 = int(tamp_list[index][0].split("_")[-1].split(".")[0])
                list_double.append((qf1, qf2, label, pred))
            if label == 0:
                qf1 = int(tamp_list[index][0].split('_')[-1].split(".")[0])
                list_single.append((qf1, label, pred))

        
        df= pd.DataFrame(list_double, columns=['qf1', 'qf2', 'label', 'pred'])
        df_count = df.groupby(['qf1', 'qf2'], as_index=False)[["label"]].count()
        df['correct'] = df['label'] * df['pred']
        df_correct = df.groupby(['qf1', 'qf2'], as_index=False)[["correct"]].sum()
        df_double = pd.concat([df_count, df_correct['correct']], axis=1)
        df_double['accuracy'] = df_double['correct'] / df_double['label']
        if epoch is None:
            df_double.to_excel(str(output_folder)+"/qf_results_double.xlsx")
        else:
            df_double.to_excel(str(output_folder)+"/qf_results_double_"+epoch+".xlsx")

        df_single= pd.DataFrame(list_single, columns=['qf1', 'label', 'pred'])
        df_count = df_single.groupby(['qf1'], as_index=False)[["label"]].count()
        df_single.loc[df_single['label'] == df_single['pred'], 'correct'] = 1
        df_single.loc[df_single['label'] != df_single['pred'], 'correct'] = 0
        df_correct = df_single.groupby(['qf1'], as_index=False)[["correct"]].sum()
        df_single = pd.concat([df_count, df_correct['correct']], axis=1) 
        df_single['accuracy'] = df_single['correct'] / df_single['label']
        if epoch is None:
            df_single.to_excel(str(output_folder)+"/qf_results_single.xlsx") 
        else:
            df_single.to_excel(str(output_folder)+"/qf_results_single_"+epoch+".xlsx")
        
        df_double = df_double.pivot("qf1", "qf2", "accuracy")
        _, ax = plt.subplots(figsize=(40,35)) 
        graphic_double = sns.heatmap(df_double, cmap="RdYlGn", annot=True, ax=ax, linewidths=0.01, linecolor='gray')
        graphic_double.invert_yaxis()
        if epoch is None:
            graphic_double.figure.savefig(str(output_folder)+"/qf_heatmap.png")
        else:
            graphic_double.figure.savefig(str(output_folder)+"/qf_heatmap_"+epoch+".png")
        

    else:
        df= pd.DataFrame(tamp_list, columns=['img_name', 'qf1', 'qf2', 'mean_IoU', 'p_mIoU', 'p_AP', 'p_f1'], dtype=float)
        df_mean = df.groupby(['qf1', 'qf2'], as_index=False)[["mean_IoU", "p_mIoU", "p_AP", "p_f1"]].mean()
        if epoch is None:
            df_mean.to_excel("/media/data/workspace/rroman/CAT-Net/qf_segmentation.xlsx")
        else:
            df_mean.to_excel("/media/data/workspace/rroman/CAT-Net/qf_segmentation_"+epoch+".xlsx")



if __name__ == '__main__':
    qf_analysis()