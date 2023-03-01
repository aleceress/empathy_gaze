import pandas as pd
from my_utils import loader
import os
import pickle
from tqdm import tqdm
from config import * 
from os.path import join 

def get_sub_features(sub_nr, dset, empathy_levels):
    fix_features, sac_features, _, _ = loader.load_event_features(f"output/new_features/{dset}/event_features_{sub_nr}.npy")
    fix_features = pd.DataFrame(fix_features)
    fix_features["empathy_level"] = empathy_levels[sub_nr]
    sac_features = pd.DataFrame(sac_features)
    sac_features["empathy_level"] = empathy_levels[sub_nr]
    return fix_features, sac_features


if __name__ == "__main__":
    AGGREGATED_FEATURES_PATH = join(OUTPUT_PATH, "aggregated_features")
    if not os.path.isdir(AGGREGATED_FEATURES_PATH):
        os.mkdir(AGGREGATED_FEATURES_PATH)
    
    if not os.path.isdir(join(AGGREGATED_FEATURES_PATH, "train")):
        os.mkdir(join(AGGREGATED_FEATURES_PATH, "train"))
    
    if not os.path.isdir(join(AGGREGATED_FEATURES_PATH, "test")):
        os.mkdir(join(AGGREGATED_FEATURES_PATH, "test"))
    
    TRAIN_PATH = join(OUTPUT_PATH, "new_features", "EyeT_OU_posterior_VI", "train")
    TEST_PATH = join(OUTPUT_PATH, "new_features", "EyeT_OU_posterior_VI", "test")

    print("Train data")
    for filename in tqdm(os.listdir(TRAIN_PATH)):
        if not os.path.isfile(join(AGGREGATED_FEATURES_PATH, "train", f"{filename.split('.')[0]}_agg.pickle")):
            with open(join(AGGREGATED_FEATURES_PATH, "train", f"{filename.split('.')[0]}_agg.pickle"), "wb") as f:
                pickle.dump([i for i in loader.load_event_features(f"{TRAIN_PATH}/{filename}")], f)
    
    print("Test data")
    for filename in tqdm(os.listdir(TEST_PATH)):
        if not os.path.isfile(join(AGGREGATED_FEATURES_PATH, "test", f"{filename.split('.')[0]}_agg.pickle")):
            with open(join(AGGREGATED_FEATURES_PATH, "test", f"{filename.split('.')[0]}_agg.pickle"), "wb") as f:
                pickle.dump([i for i in loader.load_event_features(f"{TEST_PATH}/{filename}")], f)