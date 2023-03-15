from django.urls import path
from . import views

app_name = "compare_ml"
urlpatterns = [

    path("", views.index, name="index"),
    path("logistic-regression/", views.log_model_plot, name="logistic-regression"),
    path("k-nearest-neighbours/", views.knn_model_plot, name="k-nearest-neighbours"),
    path("support-vector-machine/", views.svm_model_plot, name="support-vector-machine"),

]