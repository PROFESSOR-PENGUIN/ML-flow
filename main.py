import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import lightgbm as lgb
import sklearn.datasets as sk_data
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error

from flaml import AutoML

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env



wind_farm_data = pd.read_csv("https://github.com/dbczumar/model-registry-demo-notebook/raw/master/dataset/windfarm_data.csv", index_col=0)

def get_training_data():
  training_data = pd.DataFrame(wind_farm_data["2014-01-01":"2018-01-01"])
  X = training_data.drop(columns="power")
  y = training_data["power"]
  return X, y

def get_validation_data():
  validation_data = pd.DataFrame(wind_farm_data["2018-01-01":"2019-01-01"])
  X = validation_data.drop(columns="power")
  y = validation_data["power"]
  return X, y

X_train, y_train = get_training_data()
X_test, y_test = get_validation_data()


preprocess_pipeline=Pipeline([
    ("standardize",StandardScaler()),
])
X_train=preprocess_pipeline.fit_transform(X_train)
X_test=preprocess_pipeline.fit_transform(X_test)

mlflow.set_experiment("delete_experiment/")

# automl_settings={
#                  "time_budget":10,
#                  "task":'regression',
#                  "estimator_list":['lgbm'],
#                  }
#
# model=AutoML()

with mlflow.start_run(run_name='abc') as mlflow_run:
    model=lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    y_pred_train=model.predict(X_train)
    y_pred_test=model.predict(X_test)

    mlflow.log_metric("r2_score_train",r2_score(y_train,y_pred_train))
    mlflow.log_metric("r2_score_test",r2_score(y_test,y_pred_test))

    mlflow.log_metric("mean_sq_train", mean_squared_error(y_train, y_pred_train))
    mlflow.log_metric("mean_sq_test", mean_squared_error(y_test, y_pred_test))




