import pickle
import numpy as np
import os
import pandas as pd
from itertools import groupby
from theano import tensor as T
from operator import itemgetter
import pymc3 as pm
from config import *


def get_empathy_levels(experiment=None, type="general"):
    assert(experiment ==None or experiment == "free" or experiment =="task")
    assert(type =="general" or type == "cognitive" or type =="affective")

    questionnaires = pd.read_csv("datasets/EyeT/Questionnaire_datasetIA.csv")
    
    match experiment:
        case "free":
            questionnaires = questionnaires[questionnaires.index%2 == 0]
        case "task":
            questionnaires = questionnaires[questionnaires.index%2 == 1]
    
    match type:
        case "general":
            return questionnaires["Total Score original"]
        case "cognitive":
            cognitive_empathy_questions = [" Q25", " Q26", " Q24", " Q19", " Q27", " Q20", " Q16", " Q22", " Q15", " Q21", " Q3", " Q6", " Q5", " Q30", " Q4", " Q28", " Q1", " Q31", " Q18"]
            questionnaires = questionnaires[cognitive_empathy_questions]
            questionnaires["Total score"] = questionnaires.sum(axis = 1)
            return questionnaires["Total score"]
        case "affective":
            affective_empathy_questions = [" Q13", " Q14", " Q9", " Q8", " Q29", "  Q2", " Q11", " Q17", " Q7", " Q23", " Q10", " Q12"]
            questionnaires = questionnaires[affective_empathy_questions]
            questionnaires["Total score"] = questionnaires.sum(axis = 1)
            return questionnaires["Total score"]



def get_sub_features(sub_nr, empathy_levels, dset="test"):
    with open(
        f"{AGG_FEATURES_PATH}/{dset}/event_features_{sub_nr:02}_agg.pickle", "rb"
    ) as f:
        fix_features, sac_features, _, _ = pickle.load(f)
    fix_labels = np.repeat(empathy_levels[sub_nr], len(fix_features))
    sac_labels = np.repeat(empathy_levels[sub_nr], len(sac_features))
    return fix_features, fix_labels, sac_features, sac_labels

def get_sub(dset, experiment):
    assert(experiment == "free" or experiment == "task")
    if experiment == "free":
        filenames = [filename for filename in os.listdir(f"{AGG_FEATURES_PATH}/{dset}/") if int(filename.split("_")[2].split(".")[0])%2 == 0]
    else:
        filenames = [filename for filename in os.listdir(f"{AGG_FEATURES_PATH}/{dset}/") if int(filename.split("_")[2].split(".")[0])%2 == 1]
    return filenames

def normalize_features(features):
    features = np.array(features)
    normalized_features = (features - features.min(axis=0))/(features.max(axis=0)-features.min(axis=0))
    return normalized_features

def get_features_and_labels(dset, experiment, type="general"):

    fix_features_agg = []
    fix_labels_agg = []
    sac_features_agg = []
    sac_labels_agg = []

    filenames = get_sub(dset, experiment)

    empathy_levels = get_empathy_levels(experiment, type)

    for filename in filenames:
        sub_nr = int(filename.split("_")[2].split(".")[0])
        fix_features, fix_labels, sac_features, sac_labels = get_sub_features(sub_nr, empathy_levels, dset)
   
        for fix_feature in fix_features:
            fix_features_agg.append(fix_feature)
        for sac_feature in sac_features:
            sac_features_agg.append(sac_feature)
        for fix_label in fix_labels:
            fix_labels_agg.append(fix_label)
        for sac_label in sac_labels:
            sac_labels_agg.append(sac_label)

    fix_features_agg = normalize_features(fix_features_agg)
    fix_labels_agg = np.array(fix_labels_agg)
    sac_features_agg = normalize_features(sac_features_agg)
    sac_labels_agg = np.array(sac_labels_agg)
    return fix_features_agg, fix_labels_agg, sac_features_agg, sac_labels_agg

def get_stimuli(dset, experiment):
    fix_stimuli_agg = []
    sac_stimuli_agg = []

    filenames = get_sub(dset, experiment)

    for filename in filenames:
        sub_nr = int(filename.split("_")[2].split(".")[0])
        if sub_nr % 2 == 0:
            with open(f"{AGG_FEATURES_PATH}/test/event_features_{sub_nr:02}_agg.pickle", "rb") as f:
                _, _, fix_stimuli, sac_stimuli = pickle.load(f)
                for stim in fix_stimuli:
                    fix_stimuli_agg.append((stim[0], sub_nr))
                for stim in sac_stimuli:
                    sac_stimuli_agg.append((stim[0], sub_nr))
    return fix_stimuli_agg, sac_stimuli_agg



def combine_predictions(fix_predictions, fix_stimuli, sac_predictions, sac_stimuli):
    fix_predicted_empathy = {}
    fix_stimulus_groups = [list(group)for _, group in groupby(list(zip(fix_predictions,fix_stimuli)), itemgetter(1))]
    for fix_stimulus in fix_stimulus_groups:
        fix_predicted_empathy[(fix_stimulus[0][1][0], fix_stimulus[0][1][1])] = np.mean([value[0] for value in fix_stimulus])

    sac_predicted_empathy = {}
    sac_stimulus_groups = [list(group)for key, group in groupby(list(zip(sac_predictions, sac_stimuli)), itemgetter(1))]
    for sac_stimulus in sac_stimulus_groups:
        sac_predicted_empathy[(sac_stimulus[0][1][0], sac_stimulus[0][1][1])] = np.mean([value[0] for value in sac_stimulus])

    predicted_empathy = dict(pd.DataFrame([fix_predicted_empathy, sac_predicted_empathy]).mean())
    return predicted_empathy
    
def generate_logistic_regression_model(model_name, model_path, features, labels):
    with pm.Model() as model:
        X = pm.Data("x", features)
        y = pm.Data("y", labels)
        
        beta = pm.Normal("beta", mu=0, sigma=1, shape=features.shape[1])
        logit = T.dot(X, beta.T)
        pm.Bernoulli("empathy", pm.math.sigmoid(logit), observed=y, shape = X.eval().shape[0])
        
        trace = pm.sample(1000, random_seed=0)
        return model, trace