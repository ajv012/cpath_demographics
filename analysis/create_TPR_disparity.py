import os
from os.path import join as j_

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from tqdm import tqdm
import pdb

metric_cols = ['AUC', 'Cutoff', 'Acc',
'Y=0 P', 'Y=0 R', 'Y=0 F1', 'Y=0 Support',
'Y=1 P', 'Y=1 R', 'Y=1 F1', 'Y=1 Support',
'Macro Avg P', 'Macro Avg R', 'Macro Avg F1',
'Weight Avg P', 'Weight Avg R', 'Weight Avg F1'
]


def calc_metrics_binary(y_label, y_prob, cutoff=None, return_pred=False):
    """
    Computing (almost all) binary calculation metrics.
    
    Args:
        - y_label (np.array): (n,)-dim np.array containing ground-truth predictions.
        - y_prob (np.array): (n,)-dim np.array containing probability scores (for y=1).
        - cutoff (int): Whether to use a Yolan's J cutoff (calculated from a model)
        - return_pred (np.array): (n,)-dim np.array containing predictions using Yolan's J.
    Return:
        - results (list): List of binary classification metrics.
    """
    ### AUC
    auc = roc_auc_score(y_label, y_prob)
    
    ### Yolans J
    if cutoff == None:
        fpr, tpr, thresholds = roc_curve(y_label, y_prob)
        J = tpr - fpr
        cutoff = thresholds[np.argmax(J)]
    y_pred = np.array(y_prob > cutoff).astype(int)
    
    ### Classification Report
    out = classification_report(y_label, y_pred, output_dict=True, zero_division=0)
    if return_pred:
        return y_pred, [auc, cutoff, out['accuracy'],
            out['0.0']['precision'], out['0.0']['recall'], out['0.0']['f1-score'], out['0.0']['support'],
            out['1.0']['precision'], out['1.0']['recall'], out['1.0']['f1-score'], out['1.0']['support'],
            out['macro avg']['precision'], out['macro avg']['recall'], out['macro avg']['f1-score'],
            out['weighted avg']['precision'], out['weighted avg']['recall'], out['weighted avg']['f1-score'],
           ]
    else:
        return [auc, cutoff, out['accuracy'],
                out['0.0']['precision'], out['0.0']['recall'], out['0.0']['f1-score'], out['0.0']['support'],
                out['1.0']['precision'], out['1.0']['recall'], out['1.0']['f1-score'], out['1.0']['support'],
                out['macro avg']['precision'], out['macro avg']['recall'], out['macro avg']['f1-score'],
                out['weighted avg']['precision'], out['weighted avg']['recall'], out['weighted avg']['f1-score'],
               ]

    
def get_cv_metrics(eval_path: str, test_df: pd.DataFrame=None, col: str=None, label: str=None,
                   val_metrics: pd.DataFrame=None, seed: int=False):
    """
    Computes Cross-Validated Classification Metrics for race-stratified / overall test population.
    
    Args:
        - test_df (pd.DataFrame): DataFrame containing slide_ids with matched patient-level / slide-level information (e.g. - oncotree_code, race)
        - eval_path (str): Path to 'fold_{i}.csv' where i in {1...10}, which contains predicted probabiltiy scores for each label.
        - col (str, Optional): Column in test_df (containing categorical values) to subset the predictions by. 
        - label (str, Optional): Label to subset the test_df by.
        - val_metrics (pd.DataFrame): DataFrame that contains the Yolan's J for each fold.
        - Seed (None or int): Bootstrap seed for whether or not to resample the dataframe (used in bootstrap for loop).
        
    Return:
        - cv_metrics (pd.DataFrame): DataFrame containing classification metrics for each fold.
    """
    cv_metrics = []
    num_folds = 20
    for i in range(num_folds):

        if os.path.isfile(os.path.join(eval_path, 's_%d_checkpoint_results.pkl' % i)):
            results_df = pd_read_pickle(j_(eval_path, 's_%d_checkpoint_results.pkl' % i))  
        else:
            continue      
        
        if label is not None:
            results_df = results_df.join(test_df)
            results_df = results_df[results_df[col].str.contains(label)]
    
        if seed is not None:
            bootstrap = results_df.sample(n=results_df.shape[0], replace=True, random_state=seed).copy()
            collision = 0
            ### In case resampled df contains predictions of only one value.
            
            collision = 1
            while collision:
                bootstrap = results_df.sample(n=results_df.shape[0], replace=True, random_state=seed+1000+collision)
                y_label = np.array(bootstrap['Y'])
                y_prob = np.array(bootstrap['p_1'])
                
                ### Test for Collision (Errors when y_label or y_pred are all of one class)
                fpr, tpr, thresholds = roc_curve(y_label, y_prob)
                J = tpr - fpr
                cutoff = thresholds[np.argmax(J)]
                y_pred = np.array(y_prob > cutoff).astype(int)
                
                if (y_label.sum() == 0) or (y_label.sum() == bootstrap.shape[0]) or (y_pred.sum() == 0) or (y_pred.sum() == bootstrap.shape[0]):
                    collision += 1
                else:
                    collision = 0
            
        else:
            y_label = np.array(results_df['Y'])
            y_prob = np.array(results_df['p_1'])
        
        if val_metrics is None:
            cv_metrics.append(calc_metrics_binary(y_label, y_prob, None))
        else:
            cv_metrics.append(calc_metrics_binary(y_label, y_prob, val_metrics['Cutoff'][i]))
    
    cv_metrics = pd.DataFrame(cv_metrics)
    cv_metrics.columns = metric_cols
    cv_metrics.index.name = 'Folds'
    return cv_metrics


def get_metrics_stratified_boot(test_df, labels, col, val_metrics=None, best_fold=None, 
                                eval_path='./eval_results/EVAL_op_breast_subtype_FROM_tcga_breast_subtype_s1/', num_boot=10):
    race_cv_metrics_by_label = {}
    race_cv_FPRs_by_label = {}
    race_boot_metrics_by_label = {}
    boot_disparities_by_label = {}
    
    ### 1. Race-Stratified Evaluation
    pbar_labels = tqdm(labels, position=0, leave=True)
    for label in pbar_labels:
        pbar_labels.set_description('Bootstrap - Race %s' % label)
        
        ### 1.1. Mean of the Cross-Validated Metrics (for each race group)
        race_cv_metrics = get_cv_metrics(eval_path, test_df=test_df, col=col, 
                                    label=label, val_metrics=val_metrics, seed=None)
        race_cv_metrics = race_cv_metrics.drop(['Cutoff'], axis=1)
        race_cv_metrics_by_label[label] = race_cv_metrics
        race_cv_FPRs_by_label[label] = race_cv_metrics[['Y=0 R', 'Y=1 R']]
        
        ### 1.2. 95% Confidence Interval of the Cross-Validated Metrics (for each race group)
        race_boot_metrics = []
        for seed in tqdm(range(num_boot), total=num_boot, position=0, leave=True):
            race_boot_metrics.append(get_cv_metrics(eval_path, test_df=test_df, col=col, 
                                                    label=label, val_metrics=val_metrics, seed=seed).mean())
        race_boot_metrics = pd.concat(race_boot_metrics, axis=1).T
        race_boot_metrics.index.name = 'Runs'
        race_boot_metrics = race_boot_metrics.drop(['Cutoff'], axis=1)
        race_boot_metrics_by_label[label] = race_boot_metrics
        
    ### 2.1 Overall Evaluation
    overall_metrics = get_cv_metrics(eval_path, test_df=None, col=None, 
                                     label=None, val_metrics=val_metrics, seed=None)
    overall_metrics = overall_metrics.drop(['Cutoff'], axis=1).mean()
    
    ### 2.2 95% Confidence Interval of Overall Population Metrics
    overall_boot_metrics = []
    for seed in tqdm(range(num_boot), total=num_boot, position=0, leave=True):
        overall_boot_metrics.append(get_cv_metrics(eval_path, test_df=None, col=None,
                                                   label=None, val_metrics=val_metrics, seed=seed).mean())
    overall_boot_metrics = pd.concat(overall_boot_metrics, axis=1).T
    overall_boot_metrics.index.name = 'Runs'
    overall_boot_metrics = overall_boot_metrics.drop(['Cutoff'], axis=1)
        
    ### 3. Summary
    metrics_ci = pd.DataFrame([race_cv_metrics_by_label[label].mean().map('{:.3f}'.format).astype(str) + ' ' + \
                               race_boot_metrics_by_label[label].apply(CI_pm)
                               for label in labels])
    metrics_ci = pd.concat([metrics_ci, 
                            pd.DataFrame(overall_metrics.map('{:.3f}'.format).astype(str) + ' ' + \
                            overall_boot_metrics.apply(CI_pm)).T
                            ])
    metrics_ci.index = labels + ['Overall']
    
    ### 4. Disparities
    disparities_ci = pd.DataFrame([(race_cv_metrics_by_label[label]-overall_metrics).mean().map('{:.3f}'.format).astype(str) + ' ' + \
                                   (race_boot_metrics_by_label[label]-overall_boot_metrics).apply(CI_pm)
                                   for label in labels])
    disparities_ci.index = labels
    
    ### 5. Misses
    if best_fold is not None:
        results_df = pd_read_pickle(j_(eval_path, 's_%d_checkpoint_results.pkl' % i))
        y_label = np.array(results_df['Y'])
        y_prob = np.array(results_df['p_1'])
        y_pred, _ = calc_metrics_binary(y_label, y_prob, val_metrics['Cutoff'][i], return_pred=True)
        results_miss = results_df.copy()
        label_mapping = dict(zip([0,1], list(test_df['oncotree_code'].unique())))
        results_miss['Y_hat'] = y_pred
        results_miss = results_miss.drop(['p_0', 'p_1'], axis=1)
        results_miss.columns = ['Label', 'Prediction']
        results_miss['Label'] = results_miss['Label'].map(label_mapping)
        results_miss['Prediction'] = results_miss['Prediction'].map(label_mapping)
        results_miss['Y=%s Misdiagnosis' % label_mapping[0]] = ((results_miss['Label'] == label_mapping[0]) & (results_miss['Prediction'] == label_mapping[1])).astype(int)
        results_miss['Y=%s Misdiagnosis' % label_mapping[1]] = ((results_miss['Label'] == label_mapping[1]) & (results_miss['Prediction'] == label_mapping[0])).astype(int)
        
    return metrics_ci, disparities_ci, race_cv_FPRs_by_label, results_miss


def CI_pm(data, confidence=0.95):
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(data, p)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(data, p)
    return '(%0.3f, %0.3f)' % (lower, upper)


def stratify_summarize(eval_path, model, study='breast', y_names=['IDC', 'ILC'], best_fold=5, vel_metrics=None,  labels=['W', 'A', 'B'], col='race', lim=[-0.5, 0.5], step_size=0.1, num_boot=10, best=True): # plotting
    #eval_path='./eval_results/EVAL_op_%s_ABCD_subtype_FROM_tcga_%s_subtype_s1/' % (study, study)
    test_df = pd.read_csv('/media/hdd1/proj_fairness/code/subtype/dataset_csv/op_%s_ABCD_test_with_bins.csv' % (study), index_col=2)
    
    test_df[col] = test_df[col].astype(str)
    
    ## Calculate Results
    metrics_ci, disparities_ci, TPRs, miss = get_metrics_stratified_boot(
        test_df, labels, col, val_metrics=val_metrics, best_fold=best_fold, eval_path=eval_path, num_boot=num_boot)

    ### Create TPR Disparity
    viz_name = 'TPR_%s_%s_%s' % (study, col, model.replace(".", ""))
    create_TPR_disparity_plot(TPRs, labels=labels, y_names=y_names, name=viz_name, lim=lim, step_size=step_size, model=model, best=best)
    
    ### LaTeX Resutls
    print(
        pd.concat([metrics_ci['Y=0 R'], 
                   disparities_ci['Y=0 R'], 
                   metrics_ci['Y=1 R'], 
                   disparities_ci['Y=1 R'], 
                   metrics_ci['Macro Avg F1']], axis=1).style.to_latex())
    pdb.set_trace()
    
### Plotting Functions 
def font_prop(size = 25, fname='/media/hdd1/proj_fairness/code/subtype/post_process/assets/fonts/Arial.ttf'):
    return fm.FontProperties(size = size, fname=fname)


def configure_matplotlib(plot_args):
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = plot_args.border_thickness

    
def configure_font(label_size = 40, font='/media/hdd1/proj_fairness/code/subtype/post_process/assets/fonts/Arial.ttf'):
    font_args = dict()
    font_args['ax_label_fprop']= font_prop(label_size, font)
    font_args['ax_tick_label_fprop'] = font_prop(int(label_size * 0.75), font)
    font_args['title_fprop'] = font_prop(label_size, font)
    font_args['zoom_ax_tick_label_fprop'] = font_prop(int(label_size * 0.6), font)
    font_args['legend_fprop'] = font_prop(int(label_size * 0.65), font)
    font_args['pval_fprop'] = font_prop(int(label_size * 0.6), font)
    font_args['exptitle_fprop'] = font_prop(int(label_size * 1.5), font)
    font_args['shap_fprop'] = font_prop(int(label_size * 0.5), font)
    return font_args


def create_TPR_disparity_plot(TPRs, labels=['W', 'A', 'B'], y_names=['LUAD', 'LUSC'], name='TPR_Breast_Race', 
                              lim=[-0.5, 0.5], step_size=0.1, bound=0.1, model="CLAM", best=True):
    fig, ax = plt.subplots(figsize=(6,4), dpi=1000)

    # 10 = number of folds
    num_folds = 20
    y0_df = pd.concat([pd.concat([pd.Series([label]*num_folds), TPRs[label]['Y=0 R']], axis=1) for label in labels], axis=0)
    y1_df = pd.concat([pd.concat([pd.Series([label]*num_folds), TPRs[label]['Y=1 R']], axis=1) for label in labels], axis=0)
    y0_df.columns = ['Race', 'TPR']
    y0_df['TPR'] = y0_df['TPR'] - y0_df['TPR'].median()
    y0_df.insert(0, 'Class', y_names[0])
    y1_df.columns = ['Race', 'TPR']
    y1_df.insert(0, 'Class', y_names[1])
    y1_df['TPR'] = y1_df['TPR'] - y1_df['TPR'].median()

    combined_df = pd.concat([y0_df, y1_df])
    
    import seaborn as sns
    ax = sns.boxplot(x="Class", y="TPR", hue="Race", data=combined_df, palette=['#66c2a5', '#fc8d62', '#8da0cb'], showfliers=False)
    sns.stripplot(x="Class", y="TPR", hue="Race", data=combined_df, alpha=0.5, jitter=0.2, palette='dark:k', ax=ax, dodge=True)

    font_args = configure_font(label_size=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    #ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    plt.setp(ax.get_xticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
    plt.setp(ax.get_yticklabels(), fontproperties = font_args['ax_tick_label_fprop'])
    ax.set_ylim(lim[0]-bound, lim[1]+bound+0.001)
    ax.set_yticks(np.arange(lim[0], lim[1]+0.001, step_size))

    ### Make Legend
    ax.get_legend().remove()

    ax.set_xlabel('', fontproperties=font_args['ax_label_fprop'])
    ax.set_ylabel('TPR Race Disparity', fontproperties=font_args['ax_label_fprop'])

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    
    plt.savefig("/save/fig/path")
    plt.close()

def pd_read_pickle(path):
    results = pd.read_pickle(path)
    y_label = np.array(results['labels']).astype(float)
    
    results_df = pd.DataFrame({'Y': y_label, 
                               'p_0': np.array(results['probs'][:,0]),
                               'p_1': np.array(results['probs'][:,1])})
    results_df.index = results['slide_ids']
    results_df.index.name = 'slide_id'
    return results_df


all_models = [  
                "experiment_code",
              ]

num_folds = 20
count = 0
for model in all_models:
    
    dataroot = "results/on/tcga/validation"
        
    eval_path = j_(dataroot, model)

    metrics = []
    y_label_all, y_prob_all = [], []
    for i in range(num_folds):
        
        if os.path.isfile(os.path.join(eval_path, 'split_%d_results.pkl' % i)):
            results_df = pd_read_pickle(j_(eval_path, 'split_%d_results.pkl' % i))
        else:
            continue

        y_label = np.array(results_df['Y'])
        y_prob = np.array(results_df['p_1'])
        y_label_all.append(y_label)
        y_prob_all.append(y_prob)
        metrics.append(calc_metrics_binary(y_label, y_prob, cutoff=None))

    y_label_all = np.hstack(y_label_all)
    y_prob_all = np.hstack(y_prob_all)
    metrics.append(calc_metrics_binary(y_label_all, y_prob_all, cutoff=None))
    metrics = pd.DataFrame(metrics)
    metrics.columns = metric_cols
    val_metrics_breast = metrics.copy()
    best_fold = val_metrics_breast['AUC'].argmax()
    worst_fold = val_metrics_breast['AUC'].argmin()

    print(model)
    print('Best Fold:', best_fold)
    print('Worst Fold:', worst_fold)
    print('Cutoffs', np.array(val_metrics_breast['Cutoff']))

    dataroot = 'results/on/independent/validation'
    
    eval_path = j_(dataroot, model)

    study, y_names, best_fold, val_metrics = 'breast', ['IDC', 'ILC'], best_fold, metrics.copy()
    labels, cols = ['W', 'A', 'B'], 'race'
    
    lim, step_size = [-0.8, 0.4], 0.2 
    num_boot = 100
    print("Model = {}".format(model))
    stratify_summarize(eval_path=eval_path, model=model, study=study, y_names=y_names, best_fold=best_fold, vel_metrics=val_metrics,
                       labels=labels, col=cols, lim=lim, step_size=step_size, num_boot=num_boot, best=True)
    
    
    