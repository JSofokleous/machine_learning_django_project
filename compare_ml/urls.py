from django.urls import path
from . import views

app_name = "compare_ml"
urlpatterns = [

    path("", views.index, name="index"),
    path("log/", views.log_model_plot, name="log"),
    path("knn/", views.knn_model_plot, name="knn"),
    path("svm/", views.svm_model_plot, name="svm"),

]