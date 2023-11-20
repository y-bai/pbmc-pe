import numpy as np
import pandas as pd

import scanpy as sc
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import f1_score

from boruta import BorutaPy
from functools import partial 
from collections import OrderedDict


def get_data2(adata, cell_ids, n_hvg=2000, n_cell_pseudo=5):
    adata_ct = adata[adata.obs_names.isin(cell_ids)].copy()
    
#     sc.pp.highly_variable_genes(adata_ct,  n_top_genes=n_hvg)
#     adata_ct = adata_ct[:, adata_ct.var.highly_variable]
    
    adata_NP = adata_ct[adata_ct.obs['orig.ident'].str.startswith('NP')]
    adata_PE = adata_ct[adata_ct.obs['orig.ident'].str.startswith('PE')]
    
    df_NP = pd.DataFrame(data=adata_NP.X.toarray(), columns = adata_NP.var['highly_variable'].index)
    df_NP['orig.ident']=adata_NP.obs['orig.ident'].values
    df_PE = pd.DataFrame(data=adata_PE.X.toarray(), columns = adata_PE.var['highly_variable'].index)
    df_PE['orig.ident']=adata_PE.obs['orig.ident'].values
    
    df_NP_pseudo = None
    for x in np.unique(adata_NP.obs['orig.ident'].values):
        dataf = df_NP[df_NP['orig.ident']==x].iloc[:,:-1]
        dataff = dataf.groupby(np.arange(len(dataf))//n_cell_pseudo).mean()
        
        idx = dataff.index
        sc_cnts = dataf.groupby(np.arange(len(dataf))//n_cell_pseudo).size().values
        dataff['orig.ident']=[x+'_'+str(ix)+'_'+str(cc) for ix, cc in zip(idx, sc_cnts)]

        df_NP_pseudo = pd.concat([df_NP_pseudo, dataff],ignore_index=True)
    
    df_PE_pseudo = None
    for x in np.unique(adata_PE.obs['orig.ident'].values):
        dataf = df_PE[df_PE['orig.ident']==x].iloc[:,:-1]
        dataff = dataf.groupby(np.arange(len(dataf))//n_cell_pseudo).mean()
        idx = dataff.index
        sc_cnts = dataf.groupby(np.arange(len(dataf))//n_cell_pseudo).size().values
        dataff['orig.ident']=[x+'_'+str(ix)+'_'+str(cc) for ix, cc in zip(idx, sc_cnts)]
        
        df_PE_pseudo = pd.concat([df_PE_pseudo, dataff],ignore_index=True)
    
    df_NP_pseudo['label']=0
    df_PE_pseudo['label']=1
    
    df_pseudo = pd.concat([df_NP_pseudo, df_PE_pseudo], ignore_index=True)
    
    
    X_train, X_test, y_train, y_test = train_test_split(df_pseudo.iloc[:,:-1], 
                                                    df_pseudo.iloc[:,-1].values, 
                                                    test_size=0.3, random_state=42,
                                                   stratify=df_pseudo.iloc[:,-1].values)
    
    scaler = StandardScaler()
    scaler.fit(X_train.iloc[:,:-1])
    
    scal_mean_var = pd.DataFrame(data=np.concatenate((scaler.mean_.reshape(1,-1), scaler.var_.reshape(1,-1)), axis=0), 
                                 columns=adata_PE.var['highly_variable'].index)
    
    X_train_norm = scaler.transform(X_train.iloc[:,:-1])
    X_test_norm = scaler.transform(X_test.iloc[:,:-1])
    
    pd_X_train_norm=pd.DataFrame(data=X_train_norm, columns=list(df_pseudo.columns)[:-2])
    pd_X_test_norm=pd.DataFrame(data=X_test_norm, columns=list(df_pseudo.columns)[:-2])
    
    return pd_X_train_norm, pd_X_test_norm,y_train,y_test, X_train.iloc[:,-1],X_test.iloc[:,-1], scal_mean_var


def get_data(adata_hvg, cell_type, cell_typel='celltype_l1',  n_cell_pseudo=10):
    adata_ct = adata_hvg[adata_hvg.obs[cell_typel]==cell_type]
    adata_NP = adata_ct[adata_ct.obs['orig.ident'].str.startswith('NP')]
    adata_PE = adata_ct[adata_ct.obs['orig.ident'].str.startswith('PE')]
    
    df_NP = pd.DataFrame(data=adata_NP.X.toarray(), columns = adata_NP.var['highly_variable'].index)
    df_NP['orig.ident']=adata_NP.obs['orig.ident'].values
    df_PE = pd.DataFrame(data=adata_PE.X.toarray(), columns = adata_PE.var['highly_variable'].index)
    df_PE['orig.ident']=adata_PE.obs['orig.ident'].values
    
    df_NP_pseudo = None
    for x in np.unique(adata_NP.obs['orig.ident'].values):
        dataf = df_NP[df_NP['orig.ident']==x].iloc[:,:-1]
        dataff = dataf.groupby(np.arange(len(dataf))//n_cell_pseudo).mean()
        dataff['orig.ident']=x
        df_NP_pseudo = pd.concat([df_NP_pseudo, dataff],ignore_index=True)
    
    df_PE_pseudo = None
    for x in np.unique(adata_PE.obs['orig.ident'].values):
        dataf = df_PE[df_PE['orig.ident']==x].iloc[:,:-1]
        dataff = dataf.groupby(np.arange(len(dataf))//n_cell_pseudo).mean()
        dataff['orig.ident']=x
        df_PE_pseudo = pd.concat([df_PE_pseudo, dataff],ignore_index=True)
    
    df_NP_pseudo['label']=0
    df_PE_pseudo['label']=1
    
    df_pseudo = pd.concat([df_NP_pseudo, df_PE_pseudo], ignore_index=True)
    X_train, X_test, y_train, y_test = train_test_split(df_pseudo.iloc[:,:-1], 
                                                    df_pseudo.iloc[:,-1].values, 
                                                    test_size=0.3, random_state=42,
                                                   stratify=df_pseudo.iloc[:,-1].values)
    
    # get the number of single cells for training and testing
    
    
    scaler = StandardScaler()
    scaler.fit(X_train.iloc[:,:-1])

    X_train_norm = scaler.transform(X_train.iloc[:,:-1])
    X_test_norm = scaler.transform(X_test.iloc[:,:-1])
    
    pd_X_train_norm=pd.DataFrame(data=X_train_norm, columns=list(df_pseudo.columns)[:-2])
    pd_X_test_norm=pd.DataFrame(data=X_test_norm, columns=list(df_pseudo.columns)[:-2])
    
    return pd_X_train_norm, pd_X_test_norm,y_train,y_test, X_train.iloc[:,-1],X_test.iloc[:,-1]


def mi_sel(X, y, feature_list):
    mi_clf = partial(mutual_info_classif, random_state=42, n_neighbors=5)
    selector = SelectKBest(mi_clf,k='all').fit(X, y)
    
    # return features and corresponding mi scores in descreasing order
    feat_mi_score = dict(zip(feature_list, selector.scores_))
    return OrderedDict(sorted(feat_mi_score.items(), key=lambda item: item[1],reverse=True))

def get_class_weight(y_train):
    
    from sklearn.utils.class_weight import compute_class_weight
    cls = np.unique(y_train)
    cls_weight = compute_class_weight(class_weight='balanced', classes=cls, y=y_train)
    class_weight_dict = dict(zip(cls, cls_weight))
    return class_weight_dict

# chi-square percentile
# X is MinMaxScaler() and non-negative
def chi_perc_sel(X, y, percentile=20):
    
    from sklearn.feature_selection import SelectPercentile, chi2
    selector = SelectPercentile(chi2, percentile=percentile).fit(X, y)
    supp_ = selector.get_support()
    
    # return columns index
    return np.arange(X.shape[1])[supp_]

# Boruta selection
def boruta_sel(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from boruta import BorutaPy
    
    rf = RandomForestClassifier(n_jobs=-1, class_weight=get_class_weight(y), max_depth=5)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)
    feat_selector.fit(X, y)
    supp_ = feat_selector.support_
    
    # return columns index
    return np.arange(X.shape[1])[supp_]

# mutual-information
# X is StandardScaler()
def mi_perc_sel(X, y, percentile=20):
    
    from sklearn.feature_selection import SelectPercentile, mutual_info_classif
    mi_clf = partial(mutual_info_classif, random_state=42, n_neighbors=5)
    selector = SelectPercentile(mi_clf, percentile=percentile).fit(X, y)
    supp_ = selector.get_support()
    
    # return columns index
    return np.arange(X.shape[1])[supp_]

def feat_sel_comb(indf,y, percentile=30):
    
    bor_s = boruta_sel(indf.values, y)
    mi_sel = mi_perc_sel(indf.values, y, percentile=percentile)
    comb_feats_index = np.array(sorted(list(set.intersection(set(bor_s),set(mi_sel))))).tolist()
    
    return list(indf.columns[comb_feats_index])

def cv_train_rf(train_x, train_y):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 19)]

    # Maximum number of levels in tree
    max_depth = [int(x) for x in range(2,20)]

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    rfc = RandomForestClassifier(class_weight=get_class_weight(train_y),random_state=123)
    # Random search of parameters, using 5 fold cross validation, 
    # search across 50 different combinations, and use all available cores
    # Intuitively, precision is the ability of the classifier not to label 
    # as positive a sample that is negative, 
    # and recall is the ability of the classifier to find all the positive samples.
    scoring = {'F1_weight': 'f1_weighted', 'AUC': 'roc_auc'}
    rf_random = RandomizedSearchCV(estimator = rfc, 
                                   param_distributions = random_grid, refit='F1_weight',
                                   n_iter = 50, cv = 5, scoring = scoring,
                                   verbose=0, random_state=123, n_jobs = -1,return_train_score=True)
#     GridSearchCV(estimator = rfc, 
#                  param_grid = random_grid, cv = 5, 
#                  scoring = 'f1_weighted', n_jobs = -1,return_train_score=True)
    # re-train
    rf_random.fit(train_x, train_y)

    return rf_random


# Confusion Matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, save_fname=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    # plt.figure(figsize=(2, 3))
    fig, axa = plt.subplots(1,1,figsize=(2.5, 2.5))
    axa.grid(False)
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, fontsize=10) #, rotation=30, ha='right',rotation_mode="anchor")
        
        # xticklabels = axa.get_xticklabels()
        # for i, xticklabel in enumerate(xticklabels):
        #     xticklabel.set_y(xticklabel.get_position()[1]+0.01)
        
        plt.yticks(tick_marks, target_names,fontsize=10)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.3f}".format(cm[i, j]),
                     horizontalalignment="center", verticalalignment='center',
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",verticalalignment='center',
                     color="white" if cm[i, j] > thresh else "black",fontsize=12)


    plt.tight_layout()
    plt.ylabel('Reference',rotation=90,fontsize=10)
    plt.xlabel('Predicted',fontsize=10)
    
    plt.tick_params(direction='out', length=2, pad=1, width=1)
    
    if save_fname is not None:
        plt.savefig(save_fname, dpi=400, bbox_inches='tight')
    plt.show()
    
from sklearn.metrics import roc_curve, auc, precision_recall_curve,average_precision_score
from sklearn.metrics import recall_score,f1_score,classification_report,confusion_matrix
    
def perf_eval(best_model, test_x, test_y, cfm_norm=True):
    
    pred_prob_y = best_model.predict_proba(test_x)
    pred_y = best_model.predict(test_x)

    fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1])
    auc_x = auc(fpr,tpr)
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='AUROC=%0.4f)' % auc_x)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    ap = average_precision_score(test_y, pred_prob_y[:,1], average='weighted')

    precision, recall, thresholds = precision_recall_curve(test_y, pred_prob_y[:,1])
    plt.figure()
    plt.step(recall, precision, color='b',
         where='post', label='AUPR=%0.4f)' % ap)
#     plt.fill_between(recall, precision, step='post', alpha=0.2,
#                  color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0, 1.0])
    plt.legend(loc="lower right")


    recall = recall_score(test_y, pred_y, average='weighted')
    f1 = f1_score(test_y, pred_y, average='weighted')
    c_rprt = classification_report(test_y, pred_y)

    cnf_matrix = confusion_matrix(test_y, pred_y)
    np.set_printoptions(precision=4)

    tn, fp, fn, tp = cnf_matrix.ravel()
    
    sen = recall
    spe = tn/(tn+fp)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)
    
    return auc_x, ap, sen, spe, ppv, npv, f1
#     print('The sensitivity is: ', tp/(tp+fn))
#     print('The specificity is: ', tn/(tn+fp))
#     print('The precision or positive predictive value(PPV) is:',tp/(tp+fp))
#     print('The negative predictive value is: ',tn/(tn+fn))
#     print('The average precision score is: ', ap)
#     print('The recall score is: ', recall)
#     print('The f1 score is: ', f1)
#     print('The classification report is: \n', c_rprt)


