import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def qf_analysis(output_folder, list_data, epoch, mode):
    with open(list_data, "r") as f:
        tamp_list = [t.strip().split(',') for t in f.readlines()]

    if mode == "cls":
        list_single, list_double = [], []
        for image in tamp_list:
            label = int(image[1])
            pred = 1 if float(image[2]) >= 0.5 else 0
            if label == 1:
                qf1 = int(image[0].split('_')[-3])
                qf2 = int(image[0].split("_")[-1].split(".")[0])
                list_double.append((qf1, qf2, label, pred))
            if label == 0:
                qf1 = int(image[0].split('_')[-1].split(".")[0])
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
        

    elif mode == "seg":
        df= pd.DataFrame(tamp_list, columns=['img_name', 'qf1', 'qf2', 'mean_IoU', 'IoU'], dtype=float)
        df_mean = df.groupby(['qf1', 'qf2'], as_index=False)[["IoU"]].mean()

        df_IoU = df_mean.pivot("qf1", "qf2", "IoU")
        _, ax = plt.subplots(figsize=(30,25))
        sns.set(font_scale=2.5)
        graphic_IoU = sns.heatmap(df_IoU, cmap="RdYlGn", annot=True, ax=ax, linewidths=0.01, linecolor='gray')
        graphic_IoU.set_xticklabels(graphic_IoU.get_xmajorticklabels(), fontsize = 25)
        graphic_IoU.set_yticklabels(graphic_IoU.get_ymajorticklabels(), fontsize = 25)
        graphic_IoU.invert_yaxis()

        if epoch is None:
            df_mean.to_excel(str(output_folder)+"/qf_segmentation.xlsx")
            graphic_IoU.figure.savefig(str(output_folder)+"/qf_heatmap_IoU.png")
        else:
            df_mean.to_excel(str(output_folder)+"/qf_segmentation_"+epoch+".xlsx")
            graphic_IoU.figure.savefig(str(output_folder)+"/qf_heatmap_IoU"+epoch+".png")

    elif mode == "combined":
        with open(list_data, "r") as f:
            tamp_list = [t.strip().split(',') for t in f.readlines()]

        list_single, list_double = [], []
        for image in tamp_list:
            label = int(image[2])
            if label == 1:
                qf1 = int(image[0].split('_')[-3])
                qf2 = int(image[0].split("_")[-1].split(".")[0])
                list_double.append((qf1, qf2, image[1], label, int(image[4])))
            if label == 0:
                qf1 = int(image[0].split('_')[-1].split(".")[0])
                list_single.append((qf1, label, int(image[4])))


        if list_double != []:
            # Classification part
            df_double = pd.DataFrame(list_double, columns=['qf1', 'qf2', 'IoU', 'label', 'pred_class'], dtype=float)
            df_count = df_double.groupby(['qf1', 'qf2'], as_index=False)[["label"]].count()
            df_double['correct'] = df_double['label'] * df_double['pred_class']
            df_correct = df_double.groupby(['qf1', 'qf2'], as_index=False)[["correct"]].sum()
            df_cls = pd.concat([df_count, df_correct['correct']], axis=1)
            df_cls['accuracy'] = df_cls['correct'] / df_cls['label']
            if epoch is None:
                df_cls.to_excel(str(output_folder)+"/qf_results_cls.xlsx")
            else:
                df_cls.to_excel(str(output_folder)+"/qf_results_cls"+epoch+".xlsx")

            df_cls = df_cls.pivot('qf1', 'qf2', 'accuracy')
            _, ax = plt.subplots(figsize=(40,35))
            graphic_cls = sns.heatmap(df_cls, cmap="RdYlGn", annot=True, ax=ax, linewidths=0.01, linecolor='gray')
            graphic_cls.invert_yaxis()
            if epoch is None:
                graphic_cls.figure.savefig(str(output_folder)+"/qf_heatmap_cls.png")
            else:
                graphic_cls.figure.savefig(str(output_folder)+"/qf_heatmap_cls"+epoch+".png")

            # Segmentation part
            df_mean = df_double.groupby(['qf1', 'qf2'], as_index=False)[["IoU"]].mean()
            df_IoU = df_mean.pivot("qf1", "qf2", "IoU")
            _, ax = plt.subplots(figsize=(30,25)) 
            sns.set(font_scale=2.5)
            graphic_IoU = sns.heatmap(df_IoU, cmap="RdYlGn", annot=True, ax=ax, linewidths=0.01, linecolor='gray')
            graphic_IoU.set_xticklabels(graphic_IoU.get_xmajorticklabels(), fontsize = 25)
            graphic_IoU.set_yticklabels(graphic_IoU.get_ymajorticklabels(), fontsize = 25)
            graphic_IoU.invert_yaxis()

            if epoch is None:
                df_mean.to_excel(str(output_folder)+"/qf_segmentation.xlsx")
                graphic_IoU.figure.savefig(str(output_folder)+"/qf_heatmap_IoU.png")
            else:
                df_mean.to_excel(str(output_folder)+"/qf_segmentation_"+epoch+".xlsx")
                graphic_IoU.figure.savefig(str(output_folder)+"/qf_heatmap_IoU"+epoch+".png")

        if list_single != []:
            df_single = pd.DataFrame(list_single, columns=['qf1', 'label', 'pred_class'], dtype=float)
            df_count = df_single.groupby(['qf1'], as_index=False)[["label"]].count()
            df_single.loc[df_single['label'] == df_single['pred_class'], 'correct'] = 1
            df_single.loc[df_single['label'] != df_single['pred_class'], 'correct'] = 0
            df_correct = df_single.groupby(['qf1'], as_index=False)[["correct"]].sum()
            df_single = pd.concat([df_count, df_correct['correct']], axis=1) 
            df_single['accuracy'] = df_single['correct'] / df_single['label']

            if epoch is None:
                df_single.to_excel(str(output_folder)+"/qf_results_cls_single.xlsx") 
            else:
                df_single.to_excel(str(output_folder)+"/qf_results_cls_single_"+epoch+".xlsx")

if __name__ == '__main__':
    qf_analysis()