import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb



def prepare_features_and_target(df, threshold=0.0005):
    """
    Creates feature matrix X and binary target y from a DataFrame.
    Target = 1 if next close > current close, else 0.
    """
    df = df.copy()
    df['target'] = (df['close'].shift(-1) - df['close']) / df['close']
    df.dropna(inplace=True)

    feature_cols = [col for col in df.columns if col not in ['target', 'close']]
    X = df[feature_cols]
    y = df['target']
    return X, y

def time_series_cv_split(X, y, n_splits=5, test_size=0.1):
    """
    Expanding window generator for time series CV.
    """
    n_samples = len(X)
    test_len = int(n_samples * test_size)
    train_end = n_samples - test_len * n_splits

    for i in range(n_splits):
        train_stop = train_end + i * test_len #trainging data size increases
        test_start = train_stop #test data rolls
        test_stop = test_start + test_len 
        if test_stop > n_samples:
            break
        yield np.arange(0, train_stop), np.arange(test_start, test_stop)

def train_xgboost(X_train, y_train, params=None):
    """
    Trains XGBoost classifier on the training set with optional hyperparameters.
    """
    default_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        "n_estimators": 100,  #number of trees to build
        "learning_rate": 0.1,
        "max_depth": 3,  #max depth of each tree
        "subsample": 0.8,   #frac of training rows per tree
        "colsample_bytree": 0.8, #frac of features per tree
        "use_label_encoder": False,
        
    }

    if params: #if custom params provided 
        default_params.update(params) #override default params

    model = xgb.XGBRegressor(**default_params) #create classfier using new params
    model.fit(X_train, y_train) #fit model to training data
    return model #we know have a first version of the model, based on training 


def evaluate_model(X, y, n_splits=5, test_size=0.1, params=None):
    """
    Evaluates XGBoost on expanding CV folds.
    Plots AUC scores for performance.
    """
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(time_series_cv_split(X, y, n_splits, test_size)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        #model is run on defined params from train_xgboost
        model = train_xgboost(X_train, y_train, params=params)
        #returns predicted probabilities of class 1
        y_prob = model.predict_proba(X_test)[:, 1]
        #AUC of 1 is perfect, 0.5 is random guessing
        auc = roc_auc_score(y_test, y_prob)
        auc_scores.append(auc)



    return auc_scores
