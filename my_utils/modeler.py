import pickle
import numpy as np
import os
import pandas as pd
from itertools import groupby
from theano import tensor as T
from operator import itemgetter
import pymc3 as pm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from config import *
import arviz as az
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
    
def generate_model_ppc(model, trace):
    with model:
        ppc = pm.sample_posterior_predictive(trace)
    return az.from_pymc3(model=model, posterior_predictive=ppc) 

def generate_model_predictions(model, test_features, trace):
    with model:
        pm.set_data({"x": test_features})
        predictions = generate_model_ppc(model, trace)
        return predictions["posterior_predictive"]['empathy'].to_numpy().mean(axis=(0,1))

def make_labels_binary(labels, threshold):
    return [1 if label >= threshold else 0 for label in labels]

def generate_logistic_regression_model(model_name, features, labels, cores =4):
    model_path = join(MODELS_PATH, f"{model_name}.pickle")
    if os.path.exists(model_path):
        with open(model_path, "rb") as m:
            model_data = pickle.load(m)
            return model_data["model"], model_data["trace"]

    with pm.Model() as model:
        X = pm.Data("x", features)
        y = pm.Data("y", labels)
        
        beta = pm.Normal("beta", mu=0, sigma=10, shape=X.eval().shape[1])
        logit = T.dot(X, beta.T)
        pm.Bernoulli("empathy", pm.math.sigmoid(logit), observed=y, shape = X.eval().shape[0])
        
        trace = pm.sample(1000, tune=5000, random_seed=0, cores=cores)
        print("Saving model...")
        with open(model_path, 'wb') as m:
            pickle.dump({'model': model, 'trace': trace}, m)
        return model, trace

def generate_neg_binomial_regression_model(model_name, features, labels, cores = 2):
    model_path = join(MODELS_PATH, f"{model_name}.pickle")
    if os.path.exists(model_path):
        with open(model_path, "rb") as m:
            model_data = pickle.load(m)
            return model_data["model"], model_data["trace"]

    with pm.Model() as model:
        X = pm.Data("x", features)
        y = pm.Data("y", labels)
        a = pm.Normal("a", mu= 0, sigma=20)
        b = pm.Normal("b", mu=0, sigma=20, shape=X.eval().shape[1])

        alpha = pm.Exponential("alpha", 0.2)
        λ = pm.math.exp(a + T.dot(X, b.T))
        pm.NegativeBinomial("empathy", mu=λ, alpha=alpha, observed=y, shape = X.eval().shape[0])

        trace = pm.sample(1000, tune=7000, random_seed=0, cores=cores, target_accept = 0.9)
        print("Saving model...")
        with open(model_path, 'wb') as m:
            pickle.dump({'model': model, 'trace': trace}, m)
        return model, trace

def generate_mix_gauss_regression_model(model_name, features, labels, cores = 2, mu1 = 85, mu2 = 105):
    model_path = join(MODELS_PATH, f"{model_name}.pickle")
    if os.path.exists(model_path):
        with open(model_path, "rb") as m:
            model_data = pickle.load(m)
            return model_data["model"], model_data["trace"]

    n_components = 2
    with pm.Model() as model:
        X = pm.Data("x", features)
        y = pm.Data("y", labels)
        π = pm.Dirichlet("π", np.ones(n_components))

        α1 = pm.Normal('α1', mu=mu1, sd=5) 
        β1 = pm.Normal("β1", mu=0, sd=10, shape=X.eval().shape[1])

        α2 = pm.Normal('α2', mu=mu2, sd=5) 
        β2 = pm.Normal("β2", mu=0, sd=10, shape=X.eval().shape[1])
        
        μ1 = α1 + T.dot(X, β1.T)
        μ2 = α2 + T.dot(X, β2.T)

        σ1  = pm.HalfNormal('σ1', 10) 
        σ2  = pm.HalfNormal('σ2', 10) 
        mu = T.stack([μ1, μ2]).T
        sd = T.stack([σ1, σ2])
        
        pm.NormalMixture('empathy', π, mu, sd=sd, observed=y, shape=X.eval().shape[0])
        trace = pm.sample(1000, cores=cores, tune=3000, random_seed=0, target_accept = 0.9)
        print("Saving model...")
        with open(model_path, 'wb') as m:
            pickle.dump({'model': model, 'trace': trace}, m)
        return model, trace

def get_regression_evaluation(true, predictions):
    rmse =  mean_squared_error(true, predictions, squared= False)
    mape = mean_absolute_percentage_error(true,predictions)
    return round(rmse, 2), round(mape*100,2)