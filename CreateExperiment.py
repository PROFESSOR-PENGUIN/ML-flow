import mlflow

try:
    mlflow.create_experiment("delete_experiment/")
except:
    pass