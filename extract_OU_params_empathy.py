import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from my_utils.saver import save_event_features
from my_utils.gaze import (
    angle_between_first_and_last_points,
)
from my_utils.loader import *
import pymc3 as pm
from OrnsteinUhlenbeckPyMC.EU import Mv_EulerMaruyama
import theano.tensor as tt
from scipy.stats import iqr
import os
from os.path import join
import theano
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


lib = "pymc"
method = "SVI"
save_trace = False

DATASET_NAME = "EyeT"
DATASET_PATH = "datasets/EyeT"

fs = 120

def get_xy_features(xy, sampleRate, type_event):
    duration = xy.shape[0] / sampleRate  # calculate each event duration
    if type_event == "sac":
        angle = angle_between_first_and_last_points(xy)  # saccade direction
        ampl = np.linalg.norm(xy[0, :] - xy[-1, :])  # saccade amplitude
        return angle, ampl, duration
    else:
        return duration


# OrnsteinUhlenbeckPyMC SDE
def sde(xt, B, U, SIGMA):
    dif = U - xt
    res = tt.dot(B, dif.T)
    return res.T, SIGMA


def extract_features_sub(sub_data, sub, dset):
    """
    Extract and save the features of sub-th subject
    :param sub_data: data of the sub-th subject
    :param sub: subject index
    :param parameters: screen parameters
    :param lib: library used for the inference
    :param method: maximum a posteriori estimation or stochastic variational inference
    :return: None
    """
    
    data_np = np.random.randn(10, 2) # 10x2 data sampled from Gaussian
    data_th = theano.shared(data_np)

    with pm.Model() as model:
        print("\n\tBuilding Model...")
        # LKJ Prior over the "covariance matrix" Beta
        packed_LB = pm.LKJCholeskyCov(
            "packed_LB", n=2, eta=2, sd_dist=pm.HalfCauchy.dist(2.5)
        )  # n: dimension, eta: shape (in this case more weight on matrices with few correlations), sd_dist: distribution for std (in this case smooth)
        LB = pm.expand_packed_triangular(
            2, packed_LB
        )  # convert a packed triangular matrix into a two dimensional array
        B = pm.Deterministic("B", LB.dot(LB.T))

        U = np.zeros(2)  # prior assumed "attractor"

        # LKJ Prior over the "covariance matrix" Gamma
        packed_LS = pm.LKJCholeskyCov(
            "packed_LS", n=2, eta=2, sd_dist=pm.HalfCauchy.dist(2.5)
        )
        LS = pm.expand_packed_triangular(2, packed_LS)
        SIGMA = pm.Deterministic("SIGMA", LS.dot(LS.T))

        # Multi-variate Euler Maruyama (stochastic equation)
        X = Mv_EulerMaruyama(
            "X",
            1 / fs,  # dt
            sde,  # returns B*(U- x(t)) and Sigma when called over parameters
            (
                B,
                U,
                SIGMA,
            ),
            shape=(data_th.shape.eval()),
            testval=data_th,
            observed=data_th,
        )


    print("\nSubject number", sub)
    all_features = [] # all features of a single subject. An array in which each element is a dictionary that represents the features of each trial (avg features for fix and sac, the traces of fixations and the parameters)

    # Dividing data in sessions
    session_data = [session[1] for session in sub_data.groupby("Recording name")]

    for session, gaze_data in enumerate(session_data):
        print(f"\n\tSession number {session + 1}/{len(session_data)}")
        all_fix = []
        all_sac = []
        for _, event in gaze_data.groupby((gaze_data['Eye movement type'].shift() != gaze_data['Eye movement type']).cumsum()):
            if event["Eye movement type"].values[0]=="Fixation":
                all_fix.append(event)
            elif event["Eye movement type"].values[0]=="Saccade":
                all_sac.append(event)
         
        features = {}
        traces_fix = []
        traces_sac = []

        feature_fix = [] # all fixation subject features

        for fi, curr_fix in enumerate(all_fix):
            print(f"\tProcessing Fixation {fi + 1} of {len(all_fix)} for subject {str(sub)}")
            x_coords = np.reshape(curr_fix["Gaze point X"].values, (curr_fix["Gaze point X"].values.shape[0], 1))
            y_coords = np.reshape(curr_fix["Gaze point Y"].values, (curr_fix["Gaze point Y"].values.shape[0], 1))
            curr_fix_scanpath = np.concatenate((x_coords, y_coords), 1)
            try:
                fdur = get_xy_features(curr_fix_scanpath, fs, "fix") # duration of the fixation
                pupil_diameter_left = curr_fix["Pupil diameter left"].mean()
                if pupil_diameter_left is np.nan:
                    continue
                pupil_diameter_right = curr_fix["Pupil diameter right"].mean()
                if pupil_diameter_right is np.nan:
                    continue

                with model:
                    # Switch out the observed dataset
                    data_th.set_value(curr_fix_scanpath) # setting the fixations as observations
                    approx = pm.fit(n=20000, method=pm.ADVI(), progressbar = False, score=False) # approximate the posterior for that fixation
                    trace_fix = approx.sample(draws=10000) # sampling from the posterior
                    B_fix = trace_fix["B"].mean(axis=0) # setting as B for that fixation the mean of the samples' B
                    Sigma_fix = trace_fix["SIGMA"].mean(axis=0) # setting as Sigma for that fixation the mean of the samples' Sigma
                    B_fix_sd = iqr(trace_fix["B"], axis=0) # sd of the sampled B
                    Sigma_fix_sd = iqr(trace_fix["SIGMA"], axis=0) # sd of the sampled Sigma

            except Exception as e:
                print(str(e))
                print(
                    "\tSomething went wrong with feature extraction... Skipping fixation"
                )
                continue
  
            curr_f_fix = np.array(
                [
                    B_fix[0, 0],
                    B_fix[0, 1],
                    B_fix[1, 1],
                    B_fix_sd[0, 0],
                    B_fix_sd[0, 1],
                    B_fix_sd[1, 1],
                    Sigma_fix[0, 0],
                    Sigma_fix[0, 1],
                    Sigma_fix[1, 1],
                    Sigma_fix_sd[0, 0],
                    Sigma_fix_sd[0, 1],
                    Sigma_fix_sd[1, 1],
                    fdur,
                    pupil_diameter_left,
                    pupil_diameter_right
                ]
            ) # current fixation features 

            feature_fix.append(curr_f_fix) # appending to the fixation features

            tf = {} # dictionary containing all fixations sampled for B and S for this fixation
            tf["B"] = trace_fix["B"] # all sampled data for B for this fixation
            tf["S"] = trace_fix["SIGMA"] # all sampled data for Sigma for this fixation
            traces_fix.append(tf)
        try:
            features_fix = np.vstack(feature_fix) # stacks the fixation features vertically
        except ValueError:
            print("No valid fixations... Skipping trial")
            continue

        # does the same for saccades
        feature_sac = []
        for si, curr_sac in enumerate(all_sac):
            if len(curr_sac) < 4:
                continue
            print(f"\tProcessing Saccade {si + 1} of {len(all_sac)} for subject {sub}")
            x_coords = np.reshape(curr_sac["Gaze point X"].values, (curr_sac["Gaze point X"].values.shape[0], 1))
            y_coords = np.reshape(curr_sac["Gaze point Y"].values, (curr_sac["Gaze point Y"].values.shape[0], 1))
            curr_sac_scanpath = np.concatenate((x_coords, y_coords), 1)

            try:
                angle, ampl, sdur = get_xy_features(curr_sac_scanpath, fs, "sac")
                with model:
                    # Switch out the observed dataset
                    data_th.set_value(curr_sac_scanpath)
                    approx = pm.fit(n=20000, method=pm.ADVI(), progressbar=False, score=False)
                    trace_sac = approx.sample(draws=10000) 
                    B_sac = trace_sac["B"].mean(axis=0) 
                    Sigma_sac = trace_sac["SIGMA"].mean(axis=0) 
                    B_sac_sd = iqr(trace_sac["B"], axis=0) 
                    Sigma_sac_sd = iqr(trace_sac["SIGMA"], axis=0) 

            except Exception as e:
                print(str(e))
                print(
                    "\tSomething went wrong with feature extraction... Skipping saccade"
                )
                continue

            curr_f_sac = np.array(
                [
                    B_sac[0, 0],
                    B_sac[0, 1],
                    B_sac[1, 1],
                    B_sac_sd[0, 0],
                    B_sac_sd[0, 1],
                    B_sac_sd[1, 1],
                    Sigma_sac[0, 0],
                    Sigma_sac[0, 1],
                    Sigma_sac[1, 1],
                    Sigma_sac_sd[0, 0],
                    Sigma_sac_sd[0, 1],
                    Sigma_sac_sd[1, 1],
                    angle,
                    ampl,
                    sdur,
                ]
            )
            feature_sac.append(curr_f_sac)
            tf = {}
            tf["B"] = trace_sac["B"]
            tf["S"] = trace_sac["SIGMA"]
            traces_sac.append(tf)

        try: 
            features_sac = np.vstack(feature_sac) #one for each saccade
        except ValueError:
            print("No valid saccades... Skipping trial")
            continue

        features["label"] = float(sub)
        features["stimulus"] = session
        features["feat_fix"] = features_fix
        features["sacc_fix"] = features_sac
        features["traces_fix"] = traces_fix
        features["traces_sac"] = traces_sac

        all_features.append(features) #one for each trial

    save_event_features (
        all_features,
        DATASET_NAME,
        f"event_features_{sub:02}",
        type="OU_posterior",
        method="VI",
        dset=dset,
    )
    return f"Features saved for subject number {sub}"

def get_all_features(data, parallel=False):
    """
    Parallelized features extraction
    :param data: dataset
    :return: None
    """

    if parallel:
        n_processes = min(cpu_count(), len(data))

        with Pool(n_processes) as p:
            multiple_results = [
                p.apply_async(
                    extract_features_sub,
                    args=(
                        sub_data[:int(len(sub_data)*0.75)],
                        sub+1,
                        "train",
                    ),
                )
                for sub, sub_data in enumerate(data) if not os.path.exists(f"new_features/EyeT_OU_posterior_VI/train/event_features_{sub+1:02}.npy")
            ]
            _ = [res.get() for res in multiple_results]

        print("\n\nTest data!!\n\n")

        with Pool(n_processes) as p:
            multiple_results = [
                p.apply_async(
                    extract_features_sub,
                    args=(
                        sub_data[int(len(sub_data)*0.75):],
                        sub+1,
                        "test",
                    ),
                )
                for sub, sub_data in enumerate(data) if not os.path.exists(f"new_features/EyeT_OU_posterior_VI/test/event_features_{sub+1:02}.npy")
            ]
            _ = [res.get() for res in multiple_results]

    else:
        for sub, sub_data in enumerate(data):
            sub_nr = sub+1            
            n_train = int(len(sub_data)*0.75)

            if not os.path.exists(join("new_features", "EyeT_OU_posterior_VI", "train", f"event_features{sub_nr:02}.npy")):
                extract_features_sub(
                    sub_data[:n_train],
                    sub_nr,
                    dset="train",
                )
            if not os.path.exists(join("new_features", "EyeT_OU_posterior_VI", "test", f"event_features{sub_nr:02}.npy")):
                extract_features_sub(
                    sub_data[n_train:],
                    sub_nr,
                    dset="test",
                )


if __name__ == "__main__":
    data = load_eyeT(DATASET_PATH)
    get_all_features(data, parallel=True)

