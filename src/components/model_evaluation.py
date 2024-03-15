import os
import sys
import mlflow # 
import mlflow.sklearn
import numpy as np
import pickle
from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from src.logger.logging import logging
from src.exception. exception import customexception

class ModelEvaluation: #we make the class for model evulation
    def __init__(self):
        logging.info("evaluation started")

    def eval_metrics(self,actual,pred): #we have to avulated the matrix by passing the actual and predicted value
        rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        mae = mean_absolute_error(actual, pred)# here is MAE
        r2 = r2_score(actual, pred)# here is r3 value
        logging.info("evaluation metrics captured")
        return rmse, mae, r2

    def initiate_model_evaluation(self,train_array,test_array): #class for model_evulation
        try:
             X_test,y_test=(test_array[:,:-1], test_array[:,-1])

             model_path=os.path.join("artifacts","model.pkl") # join the model path 
             model=load_object(model_path) #load the model

             #mlflow.set_registry_uri("")
             #where you have to registry them by url
             
             logging.info("model has register") 

             tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme #tracking the url

             print(tracking_url_type_store) #print url



             with mlflow.start_run(): #start the server and make the file

                prediction=model.predict(X_test) #predictions

                (rmse,mae,r2)=self.eval_metrics(y_test,prediction)

                mlflow.log_metric("rmse", rmse)#they are logged on the files
                mlflow.log_metric("r2", r2)#they are logged on the files
                mlflow.log_metric("mae", mae)#they are logged on the files

                 # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")


        except Exception as e:
            raise customexception(e,sys)
