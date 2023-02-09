import pandas as pd
import os
import pymc3 as pm
import arviz as az
from theano import tensor as T
import numpy as np
from my_utils import modeler
import importlib
importlib.reload(modeler)
import pickle
from fastprogress import fastprogress
fastprogress.printing = lambda: True
from multiprocessing import cpu_count

n_components = 2

AGGREGATED_PATH = "output/aggregated_features/"

quest_before = pd.read_csv("datasets/EyeT/Questionnaire_datasetIA.csv")
quest_before.index.name = "Participant"
free_viewing_empathy = quest_before[quest_before.index%2 == 0]["Total Score original"]

free_fix_features_train, free_fix_labels_train, free_sac_features_train, free_sac_labels_train =  modeler.get_features(AGGREGATED_PATH, "train", "free")

with pm.Model() as free_fix_empathy:
    X = pm.Data("x", free_fix_features_train)
    y = pm.Data("y", free_fix_labels_train)

    a1 = pm.Normal("a1", mu=90, sigma=10)
    b1 = pm.Normal("b1", mu=0, sigma=10, shape = X.eval().shape[1])
    σ1 = pm.HalfNormal("σ1", sigma=1, shape = X.eval().shape[0])
    μ1 = pm.Normal("μ1", mu = a1 + T.dot(X, b1.T), sigma=σ1, shape = X.eval().shape[0])

    a2 = pm.Normal("a2", mu=105, sigma=10)
    b2 = pm.Normal("b2", mu=0, sigma=10, shape = X.eval().shape[1])
    σ2 = pm.HalfNormal("σ2", sigma=1, shape = X.eval().shape[0])
    μ2 = pm.Normal("μ2", mu = a1 + T.dot(X, b1.T), sigma=σ2, shape = X.eval().shape[0])

 
    μ = T.stack([μ1, μ2]).T
    σ = T.stack([σ1, σ2]).T
    
    weights = pm.Dirichlet("w", np.ones((X.eval().shape[0],n_components)), shape = (X.eval().shape[0],n_components))

    likelihood = pm.NormalMixture('likelihood', weights, μ,  σ, observed=y, shape=X.eval().shape[0])

    free_fix_empathy_trace = pm.sample(cores = cpu_count)

with open("models/free_fix_gaussian_mixture", 'wb') as buff:
    pickle.dump({'model': free_fix_empathy, 'trace': free_fix_empathy_trace}, buff)