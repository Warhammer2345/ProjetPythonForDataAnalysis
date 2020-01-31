from datetime import datetime
from django.http import HttpResponse, JsonResponse
import pickle
import pandas as pd

def PredictionTest(model, X_test, Y_test) :
    Y_pred = model.predict(X_test)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(Y_pred, Y_test)
    print("MSE : %f" % (mse))
    from sklearn.metrics import accuracy_score
    Y_pred = [round(x) for x in Y_pred]
    accuracy = accuracy_score(Y_pred, Y_test)
    print("Précision : %f" % (accuracy))
    return mse, accuracy    

def predictionRegressionLogistique(request):
    df = pd.read_csv("DonneeTestNettoyee.csv")
    loaded_model = pickle.load(open("modelRegressionLogistic.dat", "rb"))
    X_test = df.drop(columns=["target", "targetPrecedent"])
    Y_test = df["target"]
    mse, accuracy = PredictionTest(loaded_model, X_test, Y_test)
    return JsonResponse({"mse" : mse, "accuracy" : accuracy}, status = 200)

def predictionRandomForest(request):
    df = pd.read_csv("DonneeTestNettoyee.csv")
    loaded_model = pickle.load(open("modelRandomForest.dat", "rb"))
    X_test = df.drop(columns=["target", "targetPrecedent"])
    Y_test = df["target"]
    mse, accuracy = PredictionTest(loaded_model, X_test, Y_test)
    return JsonResponse({"mse" : mse, "accuracy" : accuracy}, status = 200)

def predictionRegressionLineaire(request) :
    df = pd.read_csv("DonneeTestNettoyee.csv")
    loaded_model = pickle.load(open("modelRegressionLineaire.dat", "rb"))
    X_test = df.drop(columns=["target", "targetPrecedent"])
    Y_test = df["target"]
    mse, accuracy = PredictionTest(loaded_model, X_test, Y_test)
    return JsonResponse({"mse" : mse, "accuracy" : accuracy}, status = 200)

def predictionXgboost(request) :
    df = pd.read_csv("DonneeTestNettoyee.csv")
    loaded_model = pickle.load(open("modelXgboost.dat", "rb"))
    X_test = df.drop(columns=["target", "targetPrecedent"])
    Y_test = df["target"]
    mse, accuracy = PredictionTest(loaded_model, X_test, Y_test)
    return JsonResponse({"mse" : mse, "accuracy" : accuracy}, status = 200)