import pickle
import numpy as np
import os
import pandas as pd

def get_sub_features(features_path, sub_nr, empathy_levels, dset="test"):
    with open(
        f"{features_path}/{dset}/event_features_{sub_nr:02}_agg.pickle", "rb"
    ) as f:
        fix_features, sac_features, _, _ = pickle.load(f)
    fix_labels = np.repeat(empathy_levels[sub_nr], len(fix_features))
    sac_labels = np.repeat(empathy_levels[sub_nr], len(sac_features))
    return fix_features, fix_labels, sac_features, sac_labels

def normalize_features(features):
    features = np.array(features)
    normalized_features = (features - features.min(axis=0))/(features.max(axis=0)-features.min(axis=0))
    return normalized_features

def get_features(path, dset, type):
    assert(type == "free" or type == "task")

    fix_features_agg = []
    fix_labels_agg = []
    sac_features_agg = []
    sac_labels_agg = []

    empathy_levels = pd.read_csv("datasets/EyeT/Questionnaire_datasetIA.csv")["Total Score original"]

    if type == "free":
        filenames = [filename for filename in os.listdir(f"{path}/{dset}/") if int(filename.split("_")[2].split(".")[0])%2 == 0]
    else:
        filenames = [filename for filename in os.listdir(f"{path}/{dset}/") if int(filename.split("_")[2].split(".")[0])%2 == 1]

    for filename in filenames:
        sub_nr = int(filename.split("_")[2].split(".")[0])
        fix_features, fix_labels, sac_features, sac_labels = get_sub_features(path, sub_nr, empathy_levels, dset)

        if sub_nr % 2 == 0 and type=="free":
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

def get_stimuli(path,dset, type):
    fix_stimuli = []
    sac_stimuli = []

    if type == "free":
        filenames = [filename for filename in os.listdir(f"{path}/{dset}/") if int(filename.split("_")[2].split(".")[0])%2 == 0]
    else:
        filenames = [filename for filename in os.listdir(f"{path}/{dset}/") if int(filename.split("_")[2].split(".")[0])%2 == 1]

    for filename in os.listdir(filenames):
        sub_nr = int(filename.split("_")[2].split(".")[0])
        if sub_nr % 2 == 0:
            with open(f"{path}/test/event_features_{sub_nr:02}_agg.pickle", "rb") as f:
                _, _, fix_stimuli, sac_stimuli = pickle.load(f)
                for stim in fix_stimuli:
                    fix_stimuli.append((stim[0], sub_nr))
                for stim in sac_stimuli:
                    sac_stimuli.append((stim[0], sub_nr))
    return fix_stimuli, sac_stimuli