# __Predicting empathy from gaze dynamics__

This project explores a possible correlation between gaze dynamics and empathy. The analysis is conducted by adapting gaze feature extraction described in [D’Amelio et al., 2023](https://github.com/phuselab/Gaze_4_behavioural_biometrics) to predict a level of empathy for subjects in the [EyeT4Empathy Dataset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9719458/).
A detailed description of the project can be found in `report.pdf`.


### __Installation__

Required libraries can be installed with Anaconda using the following commands:

```bash
conda env create --name gazeID --file environment.yml
conda activate gazeID
```

### __Gaze feature extraction__

Ornstein-Uhlenbeck features can be extracted from the dataset launching the module `extract_OU_params_empathy.py`. The dataset can be downloaded from [here](https://figshare.com/articles/dataset/Raw_Data/19209714/1) (gaze raw data) ad [here](https://figshare.com/articles/dataset/Questionnaires/19657323/2) (questionnaire data). The dataset folder must be set by adding the path to a `config.py` file, following the example of `config.example.py`. Participants raw data must be placed in a `DATASET_PATH/raw_participants/` folder, while the questionnaires can be directly inserted in `DATASET_PATH/`.
First part of the script will create a cleaned version of data, which will be saved in `DATASET_PATH/participant_cleaned/` for future usage.

Script results will be saved in `OUTPUT_PATH/features/EyeT_OU_posterior_VI/`.

The script requires a lot of time and memory to be executed. For this reason, a serialized and smaller version of the features can be already found in this repository under the path `output/aggregated features/`. Those can be derived from the computed features using the script `get_OU_aggregated_features.py`. All classification and regression code can be executed with them.


### __Regression, Classification and Results__

Results are visualized in notebooks, with a clear distinction made between free-viewing and task-oriented experiments, as well as subcategories within each empathy type (general, cognitive and affective empathy). Inside each subcategory, single fixation, saccade and aggregated predictions are shown, along with sampling diagnostics and posterior predictive checks.

- `classification_visualizations.ipynb` contains Logistic Regression experiments.
- `classification_eda.ipynb` contains a preliminary analysis for classification, based on correlation and class imbalance.
- `neg_bin_regression_visualizations.ipynb` contains Negative Binomial Regression experiments.
- `mix_gauss_regression_visualizations.ipynb` contains Gaussian Mixture Regression experiments.


