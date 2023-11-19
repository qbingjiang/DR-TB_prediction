

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")  # 忽略UserWarning兼容性警告


# ### Remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # we are interested in absolute coeff value
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics
def calculate_metric(gt, pred, threshold=0.5): 
    pred_1 = pred.copy()
    fpr, tpr, thres = roc_curve(gt, pred_1)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thres[optimal_idx]
    auc = metrics.auc(fpr, tpr) 
    pred_1[pred_1>optimal_threshold]=1
    pred_1[pred_1<1]=0 
    tn, fp, fn, tp = confusion_matrix(gt,pred_1).ravel()
    Sen = tp / float(tp+fn)
    Spe = tn / float(tn+fp)  
    return auc, Sen, Spe

def plot_SHAP(model, X_test, feature_names): 
    import shap
    # Explain the model predictions using SHAP values
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_test)
    shap_values2 = explainer(X_test) 
    # Plot the SHAP summary plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
    plt.xlabel("mean(|SHAP value|)")
    plt.savefig('relative_contributions_SHAP_DL.svg', bbox_inches='tight')

# ### Compare the performance of Logistic Regression with the different feature subsets
def run_logistic(X_train, X_test, y_train, y_test):
    # with a scaler
    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=500, tol=1e-3)
    # logit = LogisticRegression(max_iter=500, tol=1e-3)
    logit.fit(scaler.transform(X_train), y_train)
    print(logit.intercept_, logit.coef_, logit.score(scaler.transform(X_train), y_train))
    print(X_train.columns) 
    print('Train set') 

    # plot_SHAP(logit, scaler.transform(X_train), feature_names=X_train.columns) 

    pred = logit.predict_proba(scaler.transform(X_train))
    print('Logistic Regression roc-auc: {}'.format(
        calculate_metric(y_train, pred[:, 1])))
    pred_total = pred.copy()

    metrics_list = []
    metrics_list.append( calculate_metric(y_train, pred[:, 1]) )
    pred = logit.predict_proba(scaler.transform(X_test))
    pred_total = np.concatenate([pred_total, pred], axis=0)
    metrics_list.append( calculate_metric(y_test[:177], pred[:177, 1]) )
    metrics_list.append(calculate_metric(y_test[177:], pred[177:, 1]) ) 
    metrics_list = np.array(metrics_list) 
    return metrics_list, pred_total[:,1]  

def run_logistic_v2(X_train, X_test, y_train, y_test): 
    # with a scaler
    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=500, tol=1e-3)
    logit.fit(scaler.transform(X_train), y_train)
    # print(logit.intercept_, logit.coef_, logit.score(scaler.transform(X_train), y_train))
    # print(X_train.columns) 
    # print('Train set')

    pred = logit.predict_proba(scaler.transform(X_train))

    pred_total = pred.copy()

    metrics_list = []
    metrics_list.append( calculate_metric(y_train, pred[:, 1]) )

    # print('Test set')
    pred = logit.predict_proba(scaler.transform(X_test))
    pred_total = np.concatenate([pred_total, pred], axis=0)

    metrics_list.append( calculate_metric(y_test[:177], pred[:177, 1]) )
    metrics_list.append(calculate_metric(y_test[177:], pred[177:, 1]) ) 
    metrics_list = np.array(metrics_list) 
    return metrics_list, pred_total[:,1]  

def preparation(X_train, X_test, correlation_num = 0.8): 
    X_train_original = X_train.copy()
    X_test_original = X_test.copy()
    # ### Remove constant features
    constant_features = [
        feat for feat in X_train.columns if X_train[feat].std() == 0
    ]
    X_train.drop(labels=constant_features, axis=1, inplace=True)
    X_test.drop(labels=constant_features, axis=1, inplace=True)

    # ### Remove quasi-constant features
    sel = VarianceThreshold(threshold=0.01)  # 0.1 indicates 99% of observations approximately
    sel.fit(X_train)  # fit finds the features with low variance
    sum(sel.get_support()) # how many not quasi-constant?
    features_to_keep = X_train.columns[sel.get_support()]

    # remove features
    X_train = sel.transform(X_train)
    X_test = sel.transform(X_test)

    # I transform the NumPy arrays to dataframes
    X_train= pd.DataFrame(X_train)
    X_train.columns = features_to_keep
    X_test= pd.DataFrame(X_test)
    X_test.columns = features_to_keep

    # ### Remove duplicated features
    duplicated_feat = []
    for i in range(0, len(X_train.columns)):
        col_1 = X_train.columns[i]
        for col_2 in X_train.columns[i + 1:]:
            if X_train[col_1].equals(X_train[col_2]):
                duplicated_feat.append(col_2)
    # print('duplicated num: ', len(duplicated_feat) )
    X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
    X_test.drop(labels=duplicated_feat, axis=1, inplace=True)
    X_train_basic_filter = X_train.copy()
    X_test_basic_filter = X_test.copy()

    corr_features = correlation(X_train, correlation_num)
    # print('correlated features: ', len(set(corr_features)))

    X_train.drop(labels=corr_features, axis=1, inplace=True)
    X_test.drop(labels=corr_features, axis=1, inplace=True)
    X_train_corr = X_train.copy()
    X_test_corr = X_test.copy()
    return X_train, X_test

def preparation_v2(X_train, correlation_num = 0.8): 
    X_train_original = X_train.copy()
    # ### Remove constant features
    constant_features = [
        feat for feat in X_train.columns if X_train[feat].std() == 0
    ]
    X_train.drop(labels=constant_features, axis=1, inplace=True)

    # ### Remove quasi-constant features
    sel = VarianceThreshold(threshold=0.01)  # 0.1 indicates 99% of observations approximately
    sel.fit(X_train)  # fit finds the features with low variance
    sum(sel.get_support()) # how many not quasi-constant?
    features_to_keep = X_train.columns[sel.get_support()]

    # remove features
    X_train = sel.transform(X_train)

    # I transform the NumPy arrays to dataframes
    X_train= pd.DataFrame(X_train)
    X_train.columns = features_to_keep

    # ### Remove duplicated features
    duplicated_feat = []
    for i in range(0, len(X_train.columns)):
        col_1 = X_train.columns[i]
        for col_2 in X_train.columns[i + 1:]:
            if X_train[col_1].equals(X_train[col_2]):
                duplicated_feat.append(col_2)
    # print('duplicated num: ', len(duplicated_feat) )
    X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
    X_train_basic_filter = X_train.copy()
    corr_features = correlation(X_train, correlation_num)
    # print('correlated features: ', len(set(corr_features)))

    X_train.drop(labels=corr_features, axis=1, inplace=True)
    X_train_corr = X_train.copy()
    return X_train 

def simulation(data_path1, data_path2, data_path3, keepColumn=None): 
    data = pd.read_csv(data_path1)
    # separate dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(labels=['target'], axis=1),
        data['target'],
        test_size=0.3)
    X_train, X_test = preparation(X_train, X_test, correlation_num=0.9) 
    # ### Remove features using Lasso
    scaler = StandardScaler()
    scaler.fit(X_train)
    # fit a Lasso and selet features, make sure to select l1
    for i in range(1):    ##len(solver_list)
        C = 0.05
        tol = 1e-4 
        solver = 'liblinear'
        sel_ = SelectFromModel(
            LogisticRegression(tol=tol,C=C,
                            penalty='l1',
                            solver=solver
                            ))
        sel_.fit(scaler.transform(X_train), y_train) 

        # remove features with zero coefficient from dataset
        # and parse again as dataframe
        X_train_lasso = pd.DataFrame(sel_.transform(X_train)) 
        X_test_lasso = pd.DataFrame(sel_.transform(X_test)) 
        # add the columns name 
        X_train_lasso.columns = X_train.columns[(sel_.get_support())] 
        X_test_lasso.columns = X_train.columns[(sel_.get_support())] 

        features_to_keep = X_train_lasso.columns
        data_test_ext = pd.read_csv(data_path2)
        X_test_ext = data_test_ext.drop(labels=['target'], axis=1)
        y_test_ext = data_test_ext['target'] 
        # select features
        X_test_ext_anova = X_test_ext[list(features_to_keep)]
        # numpy array to dataframe
        X_test_ext_anova = pd.DataFrame(X_test_ext_anova)
        X_test_ext_anova.columns = features_to_keep

        data_test_ext3 = pd.read_csv(data_path3)
        X_test_ext3 = data_test_ext3.drop(labels=['target'], axis=1)
        y_test_ext3 = data_test_ext3['target'] 
        # select features
        X_test_ext_anova3 = X_test_ext3[list(features_to_keep)]
        # numpy array to dataframe
        X_test_ext_anova3 = pd.DataFrame(X_test_ext_anova3)
        X_test_ext_anova3.columns = features_to_keep

        # run_logistic(X_test_ext_anova,
        #             X_test_ext_anova3,
        #             y_test_ext, y_test_ext3)

        X_test_lasso_all = pd.concat([X_test_lasso, X_test_ext_anova, X_test_ext_anova3], axis=0) 
        y_test_all = pd.concat([y_test, y_test_ext, y_test_ext3], axis=0)
        # embedded methods - Lasso
        metrics_list, proba = run_logistic(X_train_lasso,
                                    X_test_lasso_all,
                                    y_train,
                                    y_test_all, 
                                    )
        return metrics_list, proba, y_train.to_list()+y_test_all.to_list(), pd.concat([X_train_lasso, X_test_lasso_all], axis=0), list(features_to_keep)

def findIndexOfFeatures(fullFeatures, selectedFeatrues): 
    ind_list = [fullFeatures.index(selectedFeatrues[i]) for i in range(len(selectedFeatrues ) ) ] 
    return ind_list 

def findTopKFeatures(fullFeatures, features_scores, topk=10): 
    features_scores_sort_ind = np.argsort(features_scores)[::-1]
    features_scores_sort = features_scores[features_scores_sort_ind] 
    features_name_sort = [fullFeatures[ features_scores_sort_ind[i] ] for i in range(len(features_scores_sort_ind))] 
    topkFeatures = [fullFeatures[ features_scores_sort_ind[i] ] for i in range(topk)] 
    return topkFeatures, features_name_sort, features_scores_sort

from sklearn.model_selection import KFold, StratifiedKFold
def simulation_featsSelection_kfold(data_path1, data_path2, data_path3, repeat_time=10, mode='DL' ): 
    data = pd.read_csv(data_path1)
    data_v2 = data.drop(labels=['target'], axis=1)
    target_v2 = data['target']
    feats_names_all = data.columns.to_list()

    data_v2 = preparation_v2(data_v2 ) 

    features_scores_list = []
    features_scores_total = np.zeros(len(feats_names_all) ) 
    for i_s in range(repeat_time): 
        kf = StratifiedKFold(n_splits=4, shuffle=True) 
        features_scores = np.zeros(len(feats_names_all) ) 
        for i_kf, [train_index, test_index] in enumerate(kf.split(data_v2, target_v2)): 
            X_train, X_test = data_v2.iloc[train_index, :], data_v2.iloc[test_index, :] 
            y_train, y_test = target_v2[train_index], target_v2[test_index] 
            # X_train, X_test = preparation(X_train, X_test) 
            
            # ### Remove features using Lasso
            scaler = StandardScaler()
            scaler.fit(X_train)

            # fit a Lasso and selet features, make sure to select l1
            for i in range(1):    ##len(solver_list)
                C = 0.05
                tol = 1e-4 
                solver = 'liblinear'
                sel_ = SelectFromModel(
                    LogisticRegression(tol=tol,C=C,
                                    penalty='l1',
                                    solver=solver
                                    ))
                sel_.fit(scaler.transform(X_train), y_train) 
                # remove features with zero coefficient from dataset
                # and parse again as dataframe
                X_train_lasso = pd.DataFrame(sel_.transform(X_train)) 
                X_test_lasso = pd.DataFrame(sel_.transform(X_test)) 
                feats_selected = X_train.columns[(sel_.get_support())].to_list() 
                ind_selected_features = findIndexOfFeatures(fullFeatures = feats_names_all, selectedFeatrues = feats_selected) 
                features_scores[ind_selected_features] = features_scores[ind_selected_features] + 1 
        features_scores_total += features_scores 
        features_scores_list.append(features_scores)
        topkFeatures, features_name_sort, features_scores_sort = findTopKFeatures(fullFeatures=feats_names_all, features_scores=features_scores, topk=10)
        # print(topkFeatures) 
    topkFeatures, features_name_sort_total, features_scores_sort_total = findTopKFeatures(fullFeatures=feats_names_all, features_scores=features_scores_total, topk=10)
    print(features_scores_sort_total[:10]) 
    print(features_name_sort_total[:10]) 

    dict = {'features_name_sort': features_name_sort_total, 
            'features_scores_sort': features_scores_sort_total 
            }  
    df = pd.DataFrame(dict) 
    df.to_csv('kfold_features_name_and_scores_{}.csv'.format(mode), encoding='utf-8-sig', index=False) 

    return features_scores_list, features_name_sort_total, features_scores_sort_total 


def simulation_exclude_featsSelection(data_path1, data_path2, data_path3, features_to_keep=None): 
    data = pd.read_csv(data_path1)
    # separate dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(labels=['target'], axis=1),
        data['target'],
        test_size=0.3
        )

    X_train, X_test = preparation(X_train, X_test, correlation_num=0.9) 

    # remove features with zero coefficient from dataset
    # and parse again as dataframe
    X_train = X_train[features_to_keep]
    X_test  = X_test[features_to_keep]
    # ### Remove features using Lasso
    scaler = StandardScaler()
    scaler.fit(X_train)

    # fit a Lasso and selet features, make sure to select l1
    for i in range(1): 

        C = 0.05
        tol = 1e-4 
        solver = 'liblinear'
        sel_ = SelectFromModel(
            LogisticRegression(tol=tol,C=C,
                            penalty='l1',
                            solver=solver
                            ))
        sel_.fit(scaler.transform(X_train), y_train) 

        # remove features with zero coefficient from dataset
        # and parse again as dataframe
        X_train_lasso = pd.DataFrame(sel_.transform(X_train)) 
        X_test_lasso = pd.DataFrame(sel_.transform(X_test)) 
        # add the columns name 
        X_train_lasso.columns = X_train.columns[(sel_.get_support())] 
        X_test_lasso.columns = X_train.columns[(sel_.get_support())] 

        features_to_keep = X_train_lasso.columns
        data_test_ext = pd.read_csv(data_path2)
        X_test_ext = data_test_ext.drop(labels=['target'], axis=1)
        y_test_ext = data_test_ext['target'] 
        # select features
        X_test_ext_anova = X_test_ext[list(features_to_keep)]
        # numpy array to dataframe
        X_test_ext_anova = pd.DataFrame(X_test_ext_anova)
        X_test_ext_anova.columns = features_to_keep

        data_test_ext3 = pd.read_csv(data_path3)
        X_test_ext3 = data_test_ext3.drop(labels=['target'], axis=1)
        y_test_ext3 = data_test_ext3['target'] 
        # select features
        X_test_ext_anova3 = X_test_ext3[list(features_to_keep)]
        # numpy array to dataframe
        X_test_ext_anova3 = pd.DataFrame(X_test_ext_anova3)
        X_test_ext_anova3.columns = features_to_keep

        # run_logistic(X_test_ext_anova,
        #             X_test_ext_anova3,
        #             y_test_ext, y_test_ext3)

        X_test_lasso_all = pd.concat([X_test_lasso, X_test_ext_anova, X_test_ext_anova3], axis=0) 
        y_test_all = pd.concat([y_test, y_test_ext, y_test_ext3], axis=0)
        # embedded methods - Lasso
        metrics_list, proba = run_logistic(X_train_lasso,
                                    X_test_lasso_all,
                                    y_train,
                                    y_test_all, 
                                    )
        return metrics_list, proba, y_train.to_list()+y_test_all.to_list(), pd.concat([X_train_lasso, X_test_lasso_all], axis=0)

def simulation_exclude_featsSelection_v2(data_path1, data_path2, data_path3, features_to_keep=None): 


    data = pd.read_csv(data_path1)
    data_v2 = data.drop(labels=['target'], axis=1)
    target_v2 = data['target']
    feats_names_all = data.columns.to_list()

    features_scores_list = []
    features_scores_total = np.zeros(len(feats_names_all) ) 

    results = []
    kf = StratifiedKFold(n_splits=4, shuffle=True ) 

    for i_kf, [train_index, test_index] in enumerate(kf.split(data_v2, target_v2)): 
        X_train, X_test = data_v2.iloc[train_index, :], data_v2.iloc[test_index, :] 
        y_train, y_test = target_v2[train_index], target_v2[test_index] 

        X_train_lasso = X_train[features_to_keep]
        X_test_lasso  = X_test[features_to_keep]

        # X_train, X_test = preparation(X_train, X_test) 
        
        # # ### Remove features using Lasso
        # scaler = StandardScaler()
        # scaler.fit(X_train_lasso)
        
        data_test_ext = pd.read_csv(data_path2)
        X_test_ext = data_test_ext.drop(labels=['target'], axis=1)
        y_test_ext = data_test_ext['target'] 
        # select features
        X_test_ext_anova = X_test_ext[list(features_to_keep)] 
        # numpy array to dataframe
        X_test_ext_anova = pd.DataFrame(X_test_ext_anova)
        X_test_ext_anova.columns = features_to_keep

        data_test_ext3 = pd.read_csv(data_path3)
        X_test_ext3 = data_test_ext3.drop(labels=['target'], axis=1)
        y_test_ext3 = data_test_ext3['target'] 
        # select features
        X_test_ext_anova3 = X_test_ext3[list(features_to_keep)]
        # numpy array to dataframe
        X_test_ext_anova3 = pd.DataFrame(X_test_ext_anova3)
        X_test_ext_anova3.columns = features_to_keep

        X_test_lasso_all = pd.concat([X_test_lasso, X_test_ext_anova, X_test_ext_anova3], axis=0) 
        y_test_all = pd.concat([y_test, y_test_ext, y_test_ext3], axis=0)
        # embedded methods - Lasso
        metrics_list, proba = run_logistic_v2(X_train_lasso,
                                    X_test_lasso_all,
                                    y_train,
                                    y_test_all
                                    )
        
        results.append([ metrics_list, proba, y_train.to_list()+y_test_all.to_list(), pd.concat([X_train_lasso, X_test_lasso_all], axis=0) ] )
    return results

def run_logistic_withoutNormalization(X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=500, tol=1e-4)
    # logit.fit(scaler.transform(X_train), y_train)
    logit.fit(X_train, y_train)
    print(logit.intercept_, logit.coef_, logit.score(X_train, y_train) )
    print(X_train.columns) 

    # plot_SHAP(logit, scaler.transform(X_train), feature_names=X_train.columns) 

    pred_train = logit.predict_proba(X_train)
    pred_total = pred_train.copy()

    metrics_list = []
    metrics_list.append( calculate_metric(y_train, pred_train[:, 1]) )

    pred = logit.predict_proba(X_test)
    pred_total = np.concatenate([pred_total, pred], axis=0)

    metrics_list.append( calculate_metric(y_test[:177], pred[:177, 1]) )

    metrics_list.append( calculate_metric(y_test[177:], pred[177:, 1]) )
    metrics_list = np.array(metrics_list)
    return pred_total[:,1], metrics_list

def run_logistic_withoutNormalization_v2(X_train, X_test, y_train, y_test):
    scaler = StandardScaler().fit(X_train)
    logit = LogisticRegression(max_iter=500, tol=1e-4)
    # logit.fit(scaler.transform(X_train), y_train)
    logit.fit(X_train, y_train)
    # print(logit.intercept_, logit.coef_, logit.score(X_train, y_train) )
    # print(X_train.columns) 

    pred_train = logit.predict_proba(X_train)
    pred_total = pred_train.copy()

    metrics_list = []
    metrics_list.append( calculate_metric(y_train, pred_train[:, 1]) )

    # print('Test set')
    # pred = logit.predict_proba(scaler.transform(X_test))
    pred = logit.predict_proba(X_test)
    pred_total = np.concatenate([pred_total, pred], axis=0)
    metrics_list.append( calculate_metric(y_test[:147], pred[:147, 1]) )
    metrics_list.append( calculate_metric(y_test[147:147+426], pred[147:147+426, 1]) )
    metrics_list.append( calculate_metric(y_test[147+426:], pred[147+426:, 1]) )
    metrics_list.append( calculate_metric(y_test[147:], pred[147:, 1]) )
    metrics_list = np.array(metrics_list)
    return pred_total[:,1], metrics_list

def run(): 
    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1_DLFeats.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2_DLFeats.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3_DLFeats.csv' 
    metrics_list_DL, proba_DL, y, x_DL, features_to_keep_DL = simulation(data_path1, data_path2, data_path3) 

    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1_radiomics.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2_radiomics.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3_radiomics.csv' 
    metrics_list_radio, proba_radio, y, x_radio, features_to_keep_radio = simulation(data_path1, data_path2, data_path3) 

    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3.csv' 

    metrics_list_selectedfrom_DL_radio, proba_from_DL_radio, y, x_from_DL_radio = simulation_exclude_featsSelection(data_path1, data_path2, data_path3, features_to_keep_DL+features_to_keep_radio)
    
    #### cal the score by the formula of logistic
    score_total = np.array([-np.log(1/proba_DL - 1) , -np.log(1/proba_radio - 1) ] ).transpose()
    score_total_df = pd.DataFrame(score_total)
    score_total_df.columns = ['score_DL', 'score_radio']
    # print(score_total.shape )
    proba_score_based, metrics_list_score_based = run_logistic_withoutNormalization(score_total_df.iloc[:411,:], X_test=score_total_df.iloc[411:,:], y_train=y[:411], y_test=y[411:])

    data = {}
    data['y'] = y 
    data['proba_DL'] = proba_DL 
    data['proba_radio'] = proba_radio 
    data['proba_score_based'] = proba_score_based 
    data['score_DL'] =  score_total[:,0]
    data['score_radio'] =  score_total[:,1]
    data['score_score_based'] =  np.array( -np.log(1/proba_score_based - 1)  ) 
    data['proba_from_DL_radio_features'] = proba_from_DL_radio 
    # data['mode'] = 'train'
    df_data = pd.DataFrame(data ) 
    df_data = df_data.reset_index()
    x_DL = x_DL.reset_index()
    x_radio = x_radio.reset_index()
    df = pd.concat([x_DL, x_radio, df_data], axis=1)
    df.to_csv('proba_table.csv', encoding='utf-8-sig', index=False) 



def run_kfold(): 

    # ## save the kfold_features_name_and_scores_DL
    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1_DLFeats.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2_DLFeats.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3_DLFeats.csv' 
    features_scores_list, features_name_sort, features_scores_sort  = simulation_featsSelection_kfold(data_path1, data_path2, data_path3, repeat_time=10, mode='DL') 
    
    # ## save kfold_features_name_and_scores_radio
    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1_radiomics.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2_radiomics.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3_radiomics.csv' 
    features_scores_list, features_name_sort, features_scores_sort  = simulation_featsSelection_kfold(data_path1, data_path2, data_path3, repeat_time=10, mode='radio') 

    features_to_keep_DL = ['DLfeat_23', 'DLfeat_113', 'DLfeat_114', 'DLfeat_181', 'DLfeat_317',
       'DLfeat_361', 'DLfeat_383', 'DLfeat_479']
    ## 40, 40, 35, 31, 29, 18, 18, 18, 

    features_to_keep_radio = ['log-sigma-3-0-mm-3D_firstorder_Maximum', 'log-sigma-3-0-mm-3D_glszm_ZoneEntropy',
                               'wavelet-HHH_firstorder_Minimum', 'wavelet-LLL_firstorder_Minimum',
                                'square_gldm_SmallDependenceHighGrayLevelEmphasis', 'lbp-3D-m1_firstorder_90Percentile', 
                                'lbp-3D-k_firstorder_Skewness']
    # 38, 37, 32, 30, 30, 29, 26

    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1_DLFeats.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2_DLFeats.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3_DLFeats.csv' 
    results_DL = simulation_exclude_featsSelection_v2(data_path1, data_path2, data_path3, features_to_keep_DL) 

    data_path1 = 'feature-selection-for-machine-learning-main/dataset_1_radiomics.csv' 
    data_path2 = 'feature-selection-for-machine-learning-main/dataset_2_radiomics.csv' 
    data_path3 = 'feature-selection-for-machine-learning-main/dataset_3_radiomics.csv' 
    results_radio = simulation_exclude_featsSelection_v2(data_path1, data_path2, data_path3, features_to_keep_radio) 

    for i in range(len(results_radio)): 
        metrics_list_DL, proba_DL, y, x_DL = results_DL[i] 
        metrics_list_radio, proba_radio, y, x_radio = results_radio[i] 

        #### cal the score by the formula of logistic
        score_total = np.array([-np.log(1/proba_DL - 1) , -np.log(1/proba_radio - 1) ] ).transpose()
        score_total_df = pd.DataFrame(score_total)
        score_total_df.columns = ['score_DL', 'score_radio']
        # print(score_total.shape )
        proba_score_based, metrics_list_score_based = run_logistic_withoutNormalization_v2(score_total_df.iloc[:441,:], X_test=score_total_df.iloc[441:,:], y_train=y[:441], y_test=y[441:])

        data = {}
        data['y'] = y 
        data['proba_DL'] = proba_DL 
        data['proba_radio'] = proba_radio 
        data['proba_score_based'] = proba_score_based 
        data['score_DL'] =  score_total[:,0]
        data['score_radio'] =  score_total[:,1]
        data['score_score_based'] =  np.array( -np.log(1/proba_score_based - 1)  ) 
        # data['mode'] = 'train'
        df_data = pd.DataFrame(data ) 
        df_data = df_data.reset_index()
        x_DL = x_DL.reset_index()
        x_radio = x_radio.reset_index()
        df = pd.concat([x_DL, x_radio, df_data], axis=1)
        df.to_csv('proba_table.csv', encoding='utf-8-sig', index=False) 



if __name__=='__main__': 
    run()  

    run_kfold()

