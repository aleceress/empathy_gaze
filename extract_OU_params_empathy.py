import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from my_utils.saver import save_event_features
from my_utils.gaze import (
    split_events,
    pixels2angles,
    angle_between_first_and_last_points,
)
from my_utils.loader import load_eyeT
import pymc3 as pm
from OrnsteinUhlenbeckPyMC.EU import Mv_EulerMaruyama
import theano.tensor as tt
from scipy.stats import iqr
import nslr_hmm

lib = "pymc"
method = "SVI"
save_trace = False

DATASET_NAME = "EyeT"
DATASET_PATH = "datasets/EyeT"

fs = 120
PARTICIPANT_DIST = pd.read_csv(
    DATASET_PATH + "/participant_distance.csv", index_col="Participant nr"
)["distance"]


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


def extract_features_sub(sub_data, sub, parameters, lib, method, dset):
    """
    Extract and save the features of sub-th subject
    :param sub_data: data of the sub-th subject
    :param sub: subject index
    :param parameters: screen parameters
    :param lib: library used for the inference
    :param method: maximum a posteriori estimation or stochastic variational inference
    :return: None
    """

    data_th = np.random.randn(10, 2) # 10x2 data sampled from Gaussian

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
            shape=(data_th.shape),
            testval=data_th,
            observed=data_th,
        )

    print("\nSubject number", sub)
    all_features = [] # all features of a single subject. An array in which each element is a dictionary that represents the features of each trial (avg features for fix and sac, the traces of fixations and the parameters)

    # Dividing data in sessions
    for session, gaze_data in enumerate(sub_data):
        print("\n\tSession number", session + 1, "/", len(sub_data))

        n_samples = gaze_data.shape[0]
        dur = n_samples / fs #n_samples/avg samples in one sec = length (in time) of the signal
        t = np.linspace(0.0, dur, n_samples) #evenly spaced numbers over the length of the signal
        gaze_data_ang = pixels2angles(
            gaze_data,
            parameters["distance"],
            parameters["width"],
            parameters["height"],
            parameters["x_res"],
            parameters["y_res"],
        )
        print("\nStarting NSLR Classification...")
        sample_class, segmentation, seg_class = nslr_hmm.classify_gaze(t, gaze_data_ang)
        print("...done. Starting CBW Estimation!")
        fixations = sample_class == nslr_hmm.FIXATION # creating a boolean vector which is 1 when the event at time t is a fixation
        sp = sample_class == nslr_hmm.SMOOTH_PURSUIT
        saccades = sample_class == nslr_hmm.SACCADE
        pso = sample_class == nslr_hmm.PSO
        fix = np.logical_or(fixations, sp).astype(
            bool
        )  # merge fixations and smooth pursuits as fixations
        sac = np.logical_or(saccades, pso).astype(bool)  # merge saccades and post saccadic oscillations as saccades

        all_fix = split_events(gaze_data, fix) # coordinates for each fixation event
        all_sac = split_events(gaze_data, sac) # coordinates for each saccade event
        
        # TO UNDERSTAND from here
        print("\tStarting CBW Estimation!")
      
        features = {}
        traces_fix = []
        traces_sac = []

        feature_fix = [] # all fixation subject features
        for fi, curr_fix in enumerate(all_fix):
            print("\tProcessing Fixation " + str(fi + 1) + " of " + str(len(all_fix)) + " for subject "+ str(sub))
            try:
                fdur = get_xy_features(curr_fix, fs, "fix") # duration of the fixation

                with model:
                    # Switch out the observed dataset
                    data_th = curr_fix # setting the fixations as observations
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
                ]
            ) # current fixation features 

            feature_fix.append(curr_f_fix) # appending to the fixation features

            tf = {} # dictionary containing all fixations sampled for B and S for this fixation
            tf["B"] = trace_fix["B"] # all sampled data for B for this fixation
            tf["S"] = trace_fix["SIGMA"] # all sampled data for Sigma for this fixation
            traces_fix.append(tf)

        features_fix = np.vstack(feature_fix) # stacks the fixation features vertically

        # does the same for saccades
        feature_sac = []
        for si, curr_sac in enumerate(all_sac):
            if len(curr_sac) < 4:
                continue
            print("\tProcessing Saccade " + str(si + 1) + " of " + str(len(all_sac)) + " for subject "+ str(sub))
            try:
                angle, ampl, sdur = get_xy_features(curr_sac, fs, "sac")
                with model:
                    # Switch out the observed dataset
                    data_th = curr_sac
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

        features_sac = np.vstack(feature_sac) #one for each saccade

        features["label"] = float(sub)
        features["stimulus"] = session
        features["feat_fix"] = features_fix 
        features["sacc_fix"] = features_sac
        features["traces_fix"] = traces_fix
        features["traces_sac"] = traces_fix

        all_features.append(features) #one for each trial

    save_event_features (
        all_features,
        DATASET_NAME,
        "event_features_" + str(sub),
        type="OU_posterior",
        method="VI",
        dset=dset,
    )

    return "Features saved for subject number " + str(sub)

def get_subject_parameters(sub):
    return {
        "distance": PARTICIPANT_DIST[sub],
        "width": 0.43,
        "height": 0.31,
        "x_res": 1080,
        "y_res": 1920,
        "fs": 120,
    }


def get_all_features(data, parallel=False):
    """
    Parallelize features extraction
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
                        (sub+1)*2,
                        get_subject_parameters((sub+1)*2),
                        lib,
                        method,
                        "train",
                    ),
                )
                for sub, sub_data in enumerate(data)
            ]
            _ = [res.get() for res in multiple_results]

        print("\n\nTest data!!\n\n")

        with Pool(n_processes) as p:
            multiple_results = [
                p.apply_async(
                    extract_features_sub,
                    args=(
                        sub_data[int(len(sub_data)*0.75):],
                        (sub+1)*2,
                        get_subject_parameters((sub+1)*2),
                        lib,
                        method,
                        "test",
                    ),
                )
                for sub, sub_data in enumerate(data)
            ]
            _ = [res.get() for res in multiple_results]
    else:

        for sub, sub_data in enumerate(data):
            sub_nr = (sub+1)*2
            n_train = int(len(sub_data)*0.75)

            extract_features_sub(
                sub_data[:n_train],
                sub_nr,
                get_subject_parameters(sub_nr),
                lib,
                method,
                dset="train",
            )
            extract_features_sub(
                sub_data[n_train:],
                sub_nr,
                get_subject_parameters(sub_nr),
                lib,
                method,
                dset="test",
            )


if __name__ == "__main__":
    data = load_eyeT(DATASET_PATH)
    get_all_features(data, parallel=True)
