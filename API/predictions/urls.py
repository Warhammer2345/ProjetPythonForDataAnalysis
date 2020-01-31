from django.urls import path
from . import views

urlpatterns = [
    path('rlin', views.predictionRegressionLineaire),
    path('rlog', views.predictionRegressionLogistique),
    path('rf', views.predictionRandomForest),
    path('xgboost', views.predictionXgboost)
]
