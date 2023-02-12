import pickle
import numpy as np
import os
import pandas as pd
from itertools import groupby
from theano import tensor as T
from operator import itemgetter
import pymc3 as pm
from config import *
from os.path import join


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
    
def generate_logistic_regression_model(model_name, features, labels):
    model_path = join(MODELS_PATH, model_name)
    if os.path.exists(model_path):
        with open(model_path, "rb") as m:
            model_data = pickle.load(m)
            return model_data["model"], model_data["trace"]
            
    with pm.Model() as model:
        X = pm.Data("x", features)
        y = pm.Data("y", labels)
        
        beta = pm.Normal("beta", mu=0, sigma=10, shape=features.shape[1])
        logit = T.dot(X, beta.T)
        pm.Bernoulli("empathy", pm.math.sigmoid(logit), observed=y, shape = X.eval().shape[0])
        
        trace = pm.sample(1000, random_seed=0)
        print("Saving model...")
        with open(join(MODELS_PATH, model_name), 'wb') as buff:
            pickle.dump({'model': model, 'trace': trace}, buff)
        return model, trace