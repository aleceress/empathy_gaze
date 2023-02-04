import pandas as pd
from my_utils import loader
import os
import pickle
from tqdm import tqdm

def get_sub_features(sub_nr, dset, empathy_levels):
    fix_features, sac_features, _, _ = loader.load_event_features(f"output/new_features/{dset}/event_features_{sub_nr}.npy")
    fix_features = pd.DataFrame(fix_features)
    fix_features["empathy_level"] = empathy_levels[sub_nr]
    sac_features = pd.DataFrame(sac_features)
    sac_features["empathy_level"] = empathy_levels[sub_nr]
    return fix_features, sac_features


if __name__ == "__main__":
    if not os.path.isdir("output/aggregated_features"):
        os.mkdir("output/aggregated_features")
    
    if not os.path.isdir("output/aggregated_features/train"):
        os.mkdir("output/aggregated_features/train")
    
    if not os.path.isdir("output/aggregated_features/test"):
        os.mkdir("output/aggregated_features/test")
    
    TRAIN_PATH = "output/new_features/EyeT_OU_posterior_VI/train"
    TEST_PATH = "output/new_features/EyeT_OU_posterior_VI/test"

    print("Train data")
    for filename in tqdm(os.listdir(TRAIN_PATH)):
        with open(f"output/aggregated_features/train/{filename.split('.')[0]}_agg.pickle", "wb") as f:
            pickle.dump([i for i in loader.load_event_features(f"{TRAIN_PATH}/{filename}")], f)

    print("Test data")
    for filename in tqdm(os.listdir(TEST_PATH)):
        with open(f"output/aggregated_features/test/{filename.split('.')[0]}_agg.pickle", "wb") as f:
            pickle.dump([i for i in loader.load_event_features(f"output/new_features/test/{filename}")], f)

    empathy_levels = pd.read_csv("datasets/EyeT/Questionnaire_datasetIA.csv")["Total Score original"]
    free_fix_features_train = pd.DataFrame()
    free_sac_features_train = pd.DataFrame()
    task_fix_features_train = pd.DataFrame()
    task_sac_features_train = pd.DataFrame()


    for filename in os.listdir("output/new_features/train"):
        sub_nr = int(filename.split("_")[2].split(".")[0])
        if sub_nr%2 == 0:
            fix_features, sac_features = get_sub_features(sub_nr, "train")
            free_fix_features_train = free_fix_features_train.append(fix_features)
            free_sac_features_train = free_sac_features_train.append(sac_features)
        else: 
            fix_features, sac_features = get_sub_features(sub_nr, "train")
            task_fix_features_train = task_fix_features_train.append(fix_features)
            task_sac_features_train = task_sac_features_train.append(sac_features)
