# from gaze import dva2pixels
import numpy as np
import scipy.io as sio
from os.path import join
import os
from os import listdir, remove
from zipfile import ZipFile
import pandas as pd
import re
from config import *
import pickle

def load_event_features(file):
    features = np.load(file, allow_pickle=True)
    n_ex = len(features) # number of trials associated with the subject

    feat_fixs = []
    feat_sacs = []
    stim_fix = []
    stim_sac = []

    for e in range(n_ex): # for each trial
        curr_data_dict = features[e]
        try:
            feat_fix = curr_data_dict["feat_fix"]  # fixation parameters for that trial
            feat_sac = curr_data_dict["sacc_fix"] # saccades parameters for that

            feat_fixs.append(feat_fix) # list of trial fixation parameters
            feat_sacs.append(feat_sac) # list of trial saccade parameters
            stim_fix.append(
                np.repeat(curr_data_dict["stimulus"], len(feat_fix))[:, np.newaxis]
            ) # saving the trial number for the fixations in an array of len number of fixations and shape [[s], [s], [s] .. [n]]
            stim_sac.append(
                np.repeat(curr_data_dict["stimulus"], len(feat_sac))[:, np.newaxis]
            ) # saving the trial number for the saccades like the previous one
        except:
            continue

    # arrays grouping the features of each trial
    feat_fixs = np.vstack(feat_fixs) # stacks all fixation features for each trial in a single array
    feat_sacs = np.vstack(feat_sacs)
    stim_fix = np.vstack(stim_fix)
    stim_sac = np.vstack(stim_sac)

    return feat_fixs, feat_sacs, stim_fix, stim_sac


def load_george_features(file):
    features = np.load(file, allow_pickle=True)
    n_ex = len(features)

    feat_fixs = []
    feat_sacs = []
    stim_fix = []
    stim_sac = []

    for e in range(n_ex):
        curr_data_dict = features[e]
        feat_fix = curr_data_dict["feat_fix"].to_numpy()
        feat_sac = curr_data_dict["sacc_fix"].to_numpy()

        feat_fixs.append(feat_fix)
        feat_sacs.append(feat_sac)
        stim_fix.append(
            np.repeat(curr_data_dict["stimulus"], len(feat_fix))[:, np.newaxis]
        )
        stim_sac.append(
            np.repeat(curr_data_dict["stimulus"], len(feat_sac))[:, np.newaxis]
        )

    feat_fixs = np.vstack(feat_fixs)
    feat_sacs = np.vstack(feat_sacs)
    stim_fix = np.vstack(stim_fix)
    stim_sac = np.vstack(stim_sac)

    return feat_fixs, feat_sacs, stim_fix, stim_sac


def load_features_file(file):
    features_dict = sio.loadmat(file)

    data = np.transpose(
        [
            features_dict["id"],
            features_dict["B00"],
            features_dict["B01"],
            features_dict["B11"],
            features_dict["S00"],
            features_dict["S01"],
            features_dict["S11"],
        ]
    )[0]

    return data


def load_features_file_new(file):
    features_dict = sio.loadmat(file)

    data = np.transpose(
        [
            features_dict["id"],
            features_dict["B00"],
            features_dict["B01"],
            features_dict["B11"],
            features_dict["S00"],
            features_dict["S01"],
            features_dict["S11"],
            features_dict["B00_s"],
            features_dict["B01_s"],
            features_dict["B11_s"],
            features_dict["S00_s"],
            features_dict["S01_s"],
            features_dict["S11_s"],
            features_dict["mu_ig"],
            features_dict["loc_ig"],
            features_dict["scale_ig"],
            features_dict["alpha"],
            features_dict["beta"],
            features_dict["kappa"],
            features_dict["loc_vm"],
            features_dict["scale_vm"],
        ]
    )[0]

    return data


def load_features_file_new_fixOnly(file):
    features_dict = sio.loadmat(file)

    data = np.transpose(
        [
            features_dict["id"],
            features_dict["B00"],
            features_dict["B01"],
            features_dict["B11"],
            features_dict["S00"],
            features_dict["S01"],
            features_dict["S11"],
            features_dict["mu_ig"],
            features_dict["loc_ig"],
            features_dict["scale_ig"],
        ]
    )[0]

    return data


def load_features(path):
    global_data = []
    for file in listdir(path):
        features_dict = sio.loadmat(join(path, file))

        data = np.transpose(
            [
                features_dict["id"],
                features_dict["B00"],
                features_dict["B01"],
                features_dict["B11"],
                features_dict["S00"],
                features_dict["S01"],
                features_dict["S11"],
            ]
        )[0]
        if len(data) < 200:  # if the dataset is coutrot
            data = data[:35]  # evens out data length
        global_data.append(data)
    return np.asarray(global_data)


def load_gazebase(
    path="datasets/GazeBase_v2_0", round="Round_9", session="S1", task="Video_1"
):
    print("Unzipping data...")
    base_path = join(path, round)
    sub_zip_list = listdir(base_path)
    try:
        for sub_zip_file in sub_zip_list:
            if sub_zip_file[0] == ".":
                continue
            curr_zip_file = join(base_path, sub_zip_file)
            with ZipFile(curr_zip_file, "r") as zipObj:
                # Extract all the contents
                new_dir = join(base_path, sub_zip_file[:-4])
                zipObj.extractall(new_dir)
                remove(curr_zip_file)
        print("Files extracted")
    except IsADirectoryError:
        print("Files already extracted")

    scanpath = []
    for sub_file in listdir(base_path):
        sub_scanpath = []
        if sub_file[0] == ".":
            continue
        sub_csv = listdir(join(base_path, sub_file, session, session + "_" + task))[0]
        eye_dataframe = pd.read_csv(
            join(base_path, sub_file, session, session + "_" + task, sub_csv)
        )
        eye_dataframe.dropna(subset=["x", "y"], inplace=True)
        raw_eye_data = eye_dataframe[["x", "y"]].to_numpy()
        # Dividing data in sessions
        sec = 1
        fs = 1000.0
        n_samples = int(sec * fs)
        for trial in range(raw_eye_data.shape[0] // (n_samples)):
            curr_gaze_ang = raw_eye_data[
                int(trial * n_samples) : int((trial + 1) * n_samples)
            ]
            # from angles to x,y coordinates
            distance = 0.55
            width = 0.474
            height = 0.297
            x_res = 1680
            y_res = 1050
            curr_gaze = dva2pixels(curr_gaze_ang, distance, width, height, x_res, y_res)
            sub_scanpath.append(curr_gaze)
        sub_scanpath = np.asarray(sub_scanpath)
        scanpath.append(sub_scanpath)
    scanpath = np.asarray(scanpath)
    parameters = {
        "distance": 0.55,
        "width": 0.474,
        "height": 0.297,
        "x_res": 1680,
        "y_res": 1050,
        "fs": 1000,
    }
    return scanpath, parameters


def load_cerf(path="datasets/CerfDataset"):
    scanpath = []
    data = sio.loadmat(join(path, "fixations.mat"))["sbj"][0]
    for subject in range(8):
        sub_scanpath = []
        for session in range(200):
            recording_x = np.concatenate(
                data[subject]["scan"][0][0][:, session][0][0][0][3]
            ).ravel()
            recording_y = np.concatenate(
                data[subject]["scan"][0][0][:, session][0][0][0][4]
            ).ravel()
            recording_x = np.reshape(recording_x, (recording_x.shape[0], 1))
            recording_y = np.reshape(recording_y, (recording_y.shape[0], 1))
            sub_scanpath.append(np.concatenate((recording_x, recording_y), 1))
        sub_scanpath = np.asarray(sub_scanpath)
        scanpath.append(sub_scanpath)
    parameters = {
        "distance": 0.8,
        "width": 0.43,
        "height": 0.31,
        "x_res": 1024,
        "y_res": 768,
        "fs": 1000.0,
    }
    scanpath = np.asarray(scanpath)
    return scanpath, parameters


def load_coutrot2(path="datasets/LondonMuseum_Ext"):
    scanpath = []
    for sub in listdir(path):
        sub_scanpath = []
        for trial in listdir(path + "/" + sub):
            data = sio.loadmat(path + "/" + sub + "/" + trial)["xy"]
            sub_scanpath.append(data)
        sub_scanpath = np.asarray(sub_scanpath)
        scanpath.append(sub_scanpath)
    parameters = {
        "distance": 0.57,
        "width": 0.4,
        "height": 0.3,
        "x_res": 1280,
        "y_res": 1024,
        "fs": 250.0,
    }
    return np.asarray(scanpath), parameters


def load_coutrot(path="datasets/LondonMuseum_Ext"):
    data_dir_x = join(path, "xcoord")
    data_dir_y = join(path, "ycoord")
    scanpath = []
    for file in listdir(data_dir_x):
        sub_scanpath = []
        # just the first one for test reasons
        subject = int(file.split("b")[1].split("_")[0])

        # load the matlab file obtaining x and y coordinates eyes positions
        sub_data_x = sio.loadmat(
            join(data_dir_x, ("new_x_sub" + str(subject) + "_xcoord.mat"))
        )["array"][0]
        sub_data_y = sio.loadmat(
            join(data_dir_y, ("new_y_sub" + str(subject) + "_ycoord.mat"))
        )["array"][0]

        for i in range(len(sub_data_x)):
            if len(sub_data_x[i]) <= 0:
                continue
            recording_x = sub_data_x[i]
            recording_y = sub_data_y[i]
            sub_scanpath.append(np.concatenate((recording_x, recording_y), 1))

        sub_scanpath = np.asarray(sub_scanpath[:35])
        scanpath.append(sub_scanpath)
    parameters = {
        "distance": 0.57,
        "width": 0.4,
        "height": 0.3,
        "x_res": 1280,
        "y_res": 1024,
        "fs": 250.0,
    }
    return np.asarray(scanpath), parameters


def load_dataset(name, path, round="Round_9", session="S1", task="Video_1"):
    if name == "FIFA":
        return load_cerf(path)
    elif name == "gazebase":
        return load_gazebase(path, round, session, task)
    elif name == "coutrot":
        return load_coutrot2(path)


def load_cerf_trial(path, sub, trial):
    mat_file = path + "/fixations.mat"
    data = sio.loadmat(mat_file)["sbj"][0]
    recording_x = np.concatenate(data[sub]["scan"][0][0][:, trial][0][0][0][3]).ravel()
    recording_y = np.concatenate(data[sub]["scan"][0][0][:, trial][0][0][0][4]).ravel()
    recording_x = np.reshape(recording_x, (recording_x.shape[0], 1))
    recording_y = np.reshape(recording_y, (recording_y.shape[0], 1))
    curr_gaze = np.concatenate((recording_x, recording_y), 1)
    return curr_gaze


def load_coutrot_trial(path, sub, trial):
    mat_file = path + sub + "/" + trial
    curr_gaze = sio.loadmat(mat_file)["xy"]
    return curr_gaze


def load_zuco_trial(path, sub, trial):
    mat_file = path + sub + "/" + trial
    data_temp = sio.loadmat(mat_file)["data"]
    curr_gaze = np.transpose(np.vstack((data_temp[:, 1], data_temp[:, 2])))
    return curr_gaze


def load_trial(dataset_name, path, sub, trial):
    if dataset_name == "coutrot":
        return load_coutrot_trial(path, sub, trial)
    elif dataset_name == "cerf":
        return load_cerf_trial(path, sub, trial)
    elif dataset_name == "zuco":
        return load_zuco_trial(path, sub, trial)
    else:
        print("Dataset not recognized")


def clean_eyeT_subject_recording(
    participant_recording_name: str, participant_data: pd.DataFrame
):
    participant_recording_cleaned = participant_data.copy()
    participant_recording_cleaned = participant_recording_cleaned[
        [
            "Gaze point X",
            "Gaze point Y",
            "Pupil diameter left",
            "Pupil diameter right",
            "Validity left",
            "Validity right",
            "Presented Stimulus name",
            "Eye movement type",
            "Recording name",
            "Recording resolution height",
            "Recording resolution width",
        ]
    ]

    participant_recording_cleaned = participant_recording_cleaned[
        (participant_recording_cleaned["Validity left"] == "Valid")
        & (participant_recording_cleaned["Validity right"] == "Valid")
        & (participant_recording_cleaned["Presented Stimulus name"].notna())
        & (
            participant_recording_cleaned["Presented Stimulus name"]
            != "Eyetracker Calibration"
        )
    ]

    participant_recording_cleaned["Pupil diameter left"] = participant_recording_cleaned["Pupil diameter left"].apply(lambda x : float(x.replace(",", ".")) if not pd.isnull(x) else x)
    participant_recording_cleaned["Pupil diameter right"] =  participant_recording_cleaned["Pupil diameter right"].apply(lambda x : float(x.replace(",", ".")) if not pd.isnull(x) else x)

    group1 = ["thumbnail_grossTrial (1)", "thumbnail_grossTrial"]

    group2 = [
        "Photo3",
        "babelia 6164137243739591",
        "Photo3 (1)",
        "Photo1 (1)",
        "Photo2 (1)",
        "Photo2",
        "Photo1",
    ]

    participant_recording_cleaned["Recording name"] = participant_recording_cleaned["Recording name"].apply(lambda x : int("".join(filter(str.isdigit,x))))
    participant_recording_cleaned.replace(group1, "grey orange", inplace=True)
    participant_recording_cleaned.replace(group2, "grey blue", inplace=True)

    if not os.path.isdir(join(DATASET_PATH, "participant_cleaned")):
        os.mkdir(join(DATASET_PATH, "participant_cleaned"))
    participant_recording_cleaned.to_csv(
        join(DATASET_PATH, "participant_cleaned", participant_recording_name)
    )
    return participant_recording_cleaned


def load_eyeT_subject_recording(participant_nr: int) -> pd.DataFrame:
    if os.path.exists(
        join(DATASET_PATH, "participant_cleaned", "Participant" + str(participant_nr).zfill(4) + ".tsv")
    ):
        participant_recording_cleaned = pd.read_csv(
            join(DATASET_PATH, "participant_cleaned", "Participant" + str(participant_nr).zfill(4) + ".tsv")
        )
    else:
        participant_recording_cleaned = clean_eyeT_subject_recording(
            "Participant" + str(participant_nr).zfill(4) + ".tsv",
            pd.read_csv(
                join(DATASET_PATH, "raw_participant", "Participant" + str(participant_nr).zfill(4) + ".tsv"),
                sep="\t",
                low_memory=False,
            )
        )
    return participant_recording_cleaned

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key = alphanum_key)

def load_eyeT() -> np.array:
    all_data = []
    print("Extracting participants data...")
    for participant_recording in sorted_nicely(os.listdir(join(DATASET_PATH, "raw_participant"))):
        participant_nr = int(participant_recording.split(".")[0][-4:])
        participant_data = load_eyeT_subject_recording(participant_nr)
        all_data.append(participant_data)
    return all_data

def load_eyeT_empathy_levels(experiment=None, type="general"):
    assert(experiment ==None or experiment == "free" or experiment =="task")
    assert(type =="general" or type == "cognitive" or type =="affective")
    questionnaires = pd.read_csv(join(DATASET_PATH, "Questionnaire_datasetIA.csv"))
    
    if experiment == "free":
         questionnaires = questionnaires[questionnaires.index%2 == 0]
    else:
        questionnaires = questionnaires[questionnaires.index%2 == 1]
    
    if type == "general":
        return questionnaires["Total Score original"]
    elif type == "cognitive":
        cognitive_empathy_questions = [" Q25", " Q26", " Q24", " Q19", " Q27", " Q20", " Q16", " Q22", " Q15", " Q21", " Q3", " Q6", " Q5", " Q30", " Q4", " Q28", " Q1", " Q31", " Q18"]
        questionnaires = questionnaires[cognitive_empathy_questions]
        questionnaires["Total score"] = questionnaires.sum(axis = 1)
        return questionnaires["Total score"]
    else:
        affective_empathy_questions = [" Q13", " Q14", " Q9", " Q8", " Q29", "  Q2", " Q11", " Q17", " Q7", " Q23", " Q10", " Q12"]
        questionnaires = questionnaires[affective_empathy_questions]
        questionnaires["Total score"] = questionnaires.sum(axis = 1)
        return questionnaires["Total score"]

def load_eyeT_sub_features(sub_nr, empathy_levels, dset="test"):
    sub_path = join(OUTPUT_PATH, "aggregated_features", dset, f"event_features_{sub_nr:02}_agg.pickle")

    with open(sub_path, "rb") as f:
        fix_features, sac_features, _, _ = pickle.load(f)

    fix_labels = np.repeat(empathy_levels[sub_nr], len(fix_features))
    sac_labels = np.repeat(empathy_levels[sub_nr], len(sac_features))
    return fix_features, fix_labels, sac_features, sac_labels

def get_eyeT_subs(dset, experiment):
    assert(experiment == "free" or experiment == "task")
    if experiment == "free":
        filenames = [filename for filename in os.listdir(join(OUTPUT_PATH, "aggregated_features", dset)) if int(filename.split("_")[2].split(".")[0])%2 == 0]
    else:
        filenames = [filename for filename in os.listdir(join(OUTPUT_PATH, "aggregated_features", dset)) if int(filename.split("_")[2].split(".")[0])%2 == 1]
    return filenames

def normalize_features(features):
    features = np.array(features)
    normalized_features = (features - features.min(axis=0))/(features.max(axis=0)-features.min(axis=0))
    return normalized_features

def get_eyeT_features_and_labels(dset, experiment, type="general", normalize=True):

    fix_features_agg = []
    fix_labels_agg = []
    sac_features_agg = []
    sac_labels_agg = []

    filenames = get_eyeT_subs(dset, experiment)

    empathy_levels = load_eyeT_empathy_levels(experiment, type)

    for filename in filenames:
        sub_nr = int(filename.split("_")[2].split(".")[0])
        fix_features, fix_labels, sac_features, sac_labels = load_eyeT_sub_features(sub_nr, empathy_levels, dset)
   
        for fix_feature in fix_features:
            fix_features_agg.append(fix_feature)
        for sac_feature in sac_features:
            sac_features_agg.append(sac_feature)
        for fix_label in fix_labels:
            fix_labels_agg.append(fix_label)
        for sac_label in sac_labels:
            sac_labels_agg.append(sac_label)

    if normalize:
        fix_features_agg = normalize_features(fix_features_agg)
        sac_features_agg = normalize_features(sac_features_agg)
    return np.array(fix_features_agg), np.array(fix_labels_agg), np.array(sac_features_agg), np.array(sac_labels_agg)

def get_stimuli(dset, experiment):
    fix_stimuli_agg = []
    sac_stimuli_agg = []

    filenames = get_eyeT_subs(dset, experiment)

    for filename in filenames:
        sub_nr = int(filename.split("_")[2].split(".")[0])
        with open(join(OUTPUT_PATH, "aggregated_features", "test", f"event_features_{sub_nr:02}_agg.pickle"), "rb") as f:
            _, _, fix_stimuli, sac_stimuli = pickle.load(f)
            for stim in fix_stimuli:
                fix_stimuli_agg.append((stim[0], sub_nr))
            for stim in sac_stimuli:
                sac_stimuli_agg.append((stim[0], sub_nr))
    return fix_stimuli_agg, sac_stimuli_agg

if __name__ == "__main__":
    data_cerf = load_cerf("../datasets/CerfDataset")
    data_coutrot = load_coutrot("../datasets/LondonMuseum_Ext")
    data_gazebase = load_gazebase("../datasets/GazeBase_v2_0")
    data_eyeT = load_eyeT("../datasets/EyeT")
