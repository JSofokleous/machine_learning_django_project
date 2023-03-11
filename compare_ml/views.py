from django.shortcuts import render
from helper_functions.fit_model import log_model

# Create your views here.
def index(request):
    return render(request, "compare_ml/index.html")

def log_model_plot(request):
    plot, prediction, prediction_prob, accuracy, f1 = log_model()
    return render(request, "compare_ml/log_model.html", {'plot': plot})

# CHOOSE BUDGET ^^ [1000, 20000]
# PICK A MODEL
# PICK FEATURES