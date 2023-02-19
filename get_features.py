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
        if not os.path.isfile(f"output/aggregated_features/train/{filename.split('.')[0]}_agg.pickle"):
            with open(f"output/aggregated_features/train/{filename.split('.')[0]}_agg.pickle", "wb") as f:
                pickle.dump([i for i in loader.load_event_features(f"{TRAIN_PATH}/{filename}")], f)

    print("Test data")
    for filename in tqdm(os.listdir(TEST_PATH)):
        if not os.path.isfile(f"output/aggregated_features/test/{filename.split('.')[0]}_agg.pickle"):
            with open(f"output/aggregated_features/test/{filename.split('.')[0]}_agg.pickle", "wb") as f:
                pickle.dump([i for i in loader.load_event_features(f"{TEST_PATH}/{filename}")], f)