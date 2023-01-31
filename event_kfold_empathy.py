from os.path import join
import os
import numpy as np
import re
from my_utils.loader import load_event_features
import pandas as pd
from os.path import join
import scipy
import os
import numpy as np
from my_utils.loader import load_event_features
from sklearn.preprocessing import label_binarize, StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, plot_confusion_matrix
import numpy_indexed as npi
from sklearn.svm import SVC
from scipy.stats import uniform
import pandas as pd
import re
from my_utils.plotter import build_roc_curve

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)


def train_sklearn(X, y, model='SVM', hyper_search=True):
    from sklearn.metrics import make_scorer, f1_score
    scorer = make_scorer(f1_score, average='macro')

    pipe_svc = make_pipeline(RobustScaler(),
                             SVC(random_state=1, C=1000, gamma=0.002, kernel='rbf'))
    distributions = dict(svc__C=scipy.stats.expon(scale=1000), svc__gamma=scipy.stats.expon(scale=.1))
        
    gs = RandomizedSearchCV(pipe_svc,
                            distributions,
                            scoring='accuracy',
                            n_iter=10,
                            n_jobs=-1,
                            cv=5)
    if hyper_search:
        gs = gs.fit(X, y)
        print('Best parameters: ', gs.best_params_)
        score = gs.score(X, y)
        print('\tAccuracy: ' + str(score))
        clf = gs.best_estimator_
        return clf
    else:
        pipe_svc = pipe_svc.fit(X, y)
        score = pipe_svc.score(X, y)
        print('\tAccuracy: ' + str(score))
        return pipe_svc


def score_fusion(clf_fix, clf_sac, X_fix_test, y_f_test, stim_f_test, X_sac_test, y_s_test, stim_s_test, model='SVM'):
    #Fixations -------
    ss = np.zeros_like(y_f_test).astype('str') # array with shape test fixation labels 
    for i in range(len(y_f_test)):
        ss[i] = str(int(y_f_test[i])) + '-' + str(int(stim_f_test[i])) # [ 'subj-stimulus', ..., ]
    if model == 'SVM':
        ppred_fix = clf_fix.decision_function(X_fix_test)
    elif model == 'GP':
        ppred_fix, _ = clf_fix.predict_y(X_fix_test) # fixation predictions for test
        #ppred_fix, _ = clf_fix.predict_f(X_fix_test)
        ppred_fix = ppred_fix.numpy()
    elif model == 'NN':
        ppred_fix = clf_fix.predict(X_fix_test)
    else:
        ppred_fix = clf_fix.predict_proba(X_fix_test)

    key, ppred_fix_comb = npi.group_by(ss).mean(ppred_fix) # mean prediction for sub-stim
    y_test = np.zeros(key.shape)
    for i,k in enumerate(key):
        l = int(k.split('-')[0]) # subject
        y_test[i] = l # prediction for sub-stim is sub j 
    
    #Saccades -------
    ss = np.zeros_like(y_s_test).astype('str')
    for i in range(len(y_s_test)):
        ss[i] = str(int(y_s_test[i])) + '-' + str(int(stim_s_test[i]))
    if model == 'SVM':
        ppred_sac = clf_sac.decision_function(X_sac_test)
    elif model == 'GP':
        ppred_sac, _ = clf_sac.predict_y(X_sac_test)
        #ppred_sac, _ = clf_sac.predict_f(X_sac_test)
        ppred_sac = ppred_sac.numpy()
    elif model == 'NN':
        ppred_sac = clf_fix.predict(X_fix_test)
    else:
        ppred_sac = clf_sac.predict_proba(X_sac_test)
    #Fusion --------
    _, ppred_sac_comb = npi.group_by(ss).mean(ppred_sac)

    ppred = np.asarray((np.matrix(ppred_fix_comb) + np.matrix(ppred_sac_comb)) / 2.) # mean of fixation and saccades
    #ppred = np.asarray((np.matrix(ppred_fix_comb)))
    y_pred = np.squeeze(np.asarray(ppred.argmax(axis=1))) # final prediction

    #import code; code.interact(local=locals())

    return y_test.astype(int), y_pred, ppred
def get_CV_splits(stim_f, yf, k):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k)
    subs_splits = []
    sub_labels = np.unique(yf)  # subject numbers
    for s in sub_labels:
        curr_stims = np.unique(stim_f[yf==s])[:,np.newaxis] #array of stimuli for each subject
        subs_splits.append(kf.split(curr_stims)) # append to a list train and test generators for each subject
    return subs_splits, sub_labels

def get_results_kfold(X_fix, yf, stim_f, X_sac, ys, stim_s, k, model='SVM', hyper_search=True, feat_type='OU'):
    sub_splits_gen, sub_labels = get_CV_splits(stim_f, yf, k=k) # get CV splits based on stimuli and subjects
    sub_splits = {}
    for i,ss in enumerate(sub_splits_gen): # for every train/test generator
        curr_splits = []
        for train_index, test_index in ss:
            curr_splits.append((train_index, test_index)) # append to curr splits the tuple of generators
        sub_splits[sub_labels[i]] = curr_splits # sets the splits of the corresponding subject in a dictionary
    acc_scores = []
    eer_scores = []
    f1_scores = []
    auc_scores = []
    for fold in range(k): # for each fold
        print('\nFold ' + str(fold+1) + ' of ' + str(k))
        train_Xf = [] # train fixations
        train_yf = [] # train fixation subjects
        train_Xs = [] # train saccades
        train_ys = [] # train saccade subects
        test_Xf = []
        test_yf = []
        test_Xs = []
        test_ys = []
        train_stf = [] # train stimuli fixations
        train_sts = [] # train stimuli saccades
        test_stf = []
        test_sts = []
        for s in sub_splits.keys():
            curr_Xf = X_fix[yf==s,:] # subject  current fixations
            curr_stf = stim_f[yf==s] # subject current fixation stimuli
            curr_Xs = X_sac[ys==s,:] # subject current saccades
            curr_sts = stim_s[ys==s] # subject current saccade stimuli
            train_index = sub_splits[s][fold][0] # fold-number-th train stimulus of the subject
            test_index = sub_splits[s][fold][1] # fold-number-th test stimulus of the subject
            for ti in train_index:
                train_Xf.append(curr_Xf[curr_stf==ti]) # append to train fixations the fixations where the stimulus belongs to train in that fold 
                train_stf.append(curr_stf[curr_stf==ti]) # same for stimuli
                train_yf.append(np.repeat(s, len(train_stf[-1]))) # subject number with the same shape of stimuli (to label them)
                train_Xs.append(curr_Xs[curr_sts==ti])
                train_sts.append(curr_sts[curr_sts==ti])
                train_ys.append(np.repeat(s, len(train_sts[-1])))
            for ti in test_index:
                test_Xf.append(curr_Xf[curr_stf==ti])
                test_stf.append(curr_stf[curr_stf==ti])
                test_yf.append(np.repeat(s, len(test_stf[-1])))
                test_Xs.append(curr_Xs[curr_sts==ti])
                test_sts.append(curr_sts[curr_sts==ti])
                test_ys.append(np.repeat(s, len(test_sts[-1])))
        train_Xf = np.vstack(train_Xf) # array of train fixations
        train_yf = np.concatenate(train_yf) # array of train fixation subjects (labels)
        train_stf = np.concatenate(train_stf)
        train_Xs = np.vstack(train_Xs)
        train_ys = np.concatenate(train_ys)
        train_sts = np.concatenate(train_sts)
        test_Xf = np.vstack(test_Xf)
        test_yf = np.concatenate(test_yf)
        test_stf = np.concatenate(test_stf)
        test_Xs = np.vstack(test_Xs)
        test_ys = np.concatenate(test_ys)
        test_sts = np.concatenate(test_sts)

        print('\nTraining Fixations (SVM)')
        clf_fix = train_sklearn(train_Xf, train_yf, model=model, hyper_search=hyper_search) # parameters for fixations
        print('Training Saccades (SVM)')
        clf_sac = train_sklearn(train_Xs, train_ys, model=model, hyper_search=hyper_search) # parameters for saccades

        y_test, y_pred_test, y_score = score_fusion(clf_fix, clf_sac, 
                                                    test_Xf, 
                                                    test_yf, test_stf, 
                                                    test_Xs, 
                                                    test_ys, test_sts, 
                                                    model=model)

        f1score = f1_score(y_true=y_test, y_pred=y_pred_test, average='macro')
        acc_score = accuracy_score(y_true=y_test, y_pred=y_pred_test)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        auc, eer, _, _ = build_roc_curve(y_test_bin, y_score, max(np.unique(y_test))+1, None, None, show=False)

        print('\nTest Accuracy Score: ' + str(f1score))
        acc_scores.append(acc_score)
        f1_scores.append(f1score) 
        auc_scores.append(auc)
        eer_scores.append(eer)

    return np.mean(acc_scores), np.std(acc_scores), np.mean(f1_scores), np.std(f1_scores), np.mean(auc_scores), np.std(auc_scores), np.mean(eer_scores), np.std(eer_scores)


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)

def load_dataset(path, nsub=None, num_sessions=None):
    global_data_fix = []
    global_data_sac = []
    subs = sorted_nicely(os.listdir(path))
    if nsub is not None:
        subs = subs[:nsub]
    subs_considered = 0
    for file in subs: # for every subject
        if file == '.DS_Store':
            continue
        
        # arrays grouping the features of each trial
        fix_data, sac_data, stim_fix, stim_sac = load_event_features(join(path, file))
    
        if num_sessions is not None:
            ns = len(np.unique(stim_fix))
            if ns < num_sessions:
                continue
        label = int(file.split("_")[2].split(".")[0]) # the subject number     
        curr_label_f = np.ones([fix_data.shape[0], 1]) * label 
        curr_label_s = np.ones([sac_data.shape[0], 1]) * label
        fix_data = np.hstack([curr_label_f, stim_fix, fix_data]) # add as first two numbers of the fixation array the subject and the stimulus
        sac_data = np.hstack([curr_label_s, stim_sac, sac_data]) # same for list of saccades
        global_data_fix.append(fix_data) 
        global_data_sac.append(sac_data)
        subs_considered += 1
    data_fix = np.vstack(global_data_fix) # puts in the same list labeled features from every trial
    data_sac = np.vstack(global_data_sac)
    print('\nLoaded ' + str(subs_considered) + ' subjects...')
    return data_fix, data_sac

def generate_empathy_levels():
    empathy_questionnaire_data = pd.read_csv("datasets/EyeT/Questionnaire_datasetIA.csv")
    empathy_questionnaire_data.index.name = "Participant"
    empathy_questionnaire_free = empathy_questionnaire_data[empathy_questionnaire_data.index%2 == 0]
    empathy_scores = empathy_questionnaire_free["Total Score original"]
    low_upper_bound = np.percentile(empathy_scores, 33)
    high_lower_bound = np.percentile(empathy_scores, 66)
    empathy_level = {}
    for subject, _ in empathy_scores.items():
        if empathy_scores[subject] < low_upper_bound:
            empathy_level[subject] = "low"
        elif empathy_scores[subject] > high_lower_bound:
            empathy_level[subject] = "high"
        else:
            empathy_level[subject] = "medium"
    return empathy_level



if __name__ == "__main__":
    dataset_name = "EyeT_OU_posterior_VI"
    feat_type = 'OU'
    model = 'SVM'

    print('\n\tEyeT Dataset (OU features)...\n')
    train_dir = join(join('new_features', dataset_name), 'train')
    test_dir = join(join('new_features', dataset_name), 'test') 

    data_fix_train, data_sac_train = load_dataset(train_dir) # arrays of labeled data (first two elements are subject and stimulus) for the train set
    data_fix_test, data_sac_test = load_dataset(test_dir)
    data_fix = np.vstack([data_fix_train, data_fix_test]) # labeled fixation arrays for test and train set
    data_sac = np.vstack([data_sac_train, data_sac_test]) # labeled saccade arrays for test and train set

    empathy_levels = generate_empathy_levels()

    fix_coords = data_fix[:, 2:] # fixations
    fix_subject_numbers = data_fix[:, 0] # subject numbers
    fix_labels = [empathy_levels[i] for i in fix_subject_numbers]
    stim_f = data_fix[:, 1] # stimulus numbers

    sac_coords = data_sac[:, 2:] # saccades
    sac_subject_numbers = data_sac[:, 0] # subject numbers
    sac_labels = [empathy_levels[i] for i in sac_subject_numbers]
    stim_s = data_sac[:, 1]

    unique_f, counts_f = np.unique(fix_labels, return_counts=True) 
    cf = dict(zip(unique_f, counts_f)) # dictionary containing the number of fixations belonging to each empathy level

    unique_s, counts_s = np.unique(sac_labels, return_counts=True)
    cs = dict(zip(unique_s, counts_s)) # dictionary containing the number of saccades belonging to each empathy level

    print('\n-------------------------------')
    print('\nFixations Counts per Class: \n' + str(cf))
    print(' ')
    print('Saccades Counts per Class: \n' + str(cs))
    print('\n-------------------------------')

    acc_score, acc_std, f1_score, f1_std, auc_score, auc_std, eer_score, eer_std = get_results_kfold(fix_coords, fix_labels, stim_f, sac_coords, sac_labels, stim_s, k=10, model=model, hyper_search=True, feat_type=feat_type)

    print('\nAccuracy CV score: ' + str(acc_score) + ' +- ' + str(acc_std))
    print('F1 CV score: ' + str(f1_score) + ' +- ' + str(f1_std))
    print('AUC CV score: ' + str(auc_score) + ' +- ' + str(auc_std))
    print('EER CV score: ' + str(eer_score) + ' +- ' + str(eer_std))
    print(' ')