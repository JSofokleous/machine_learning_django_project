from django.shortcuts import render
from django import forms
from helper_functions.fit_model import log_model, knn_model

# Create your views here.

# DEFAULT VIEW
def index(request):
    return render(request, "compare_ml/index.html")

# LOGARITHMIC REGRESSION ML MODEL VIEW
def log_model_plot(request):
    # Get plot using default rent budget/property features
    plot, prediction, prediction_prob, accuracy, f1 = log_model(5000, [940, 5, 1, 0, 0])
    # plot, prediction, prediction_prob, accuracy, f1 = log_model(5000, [940, 5, 1, 0, 0, 0, 0, 0 ,0])

    # Maintain input values upon submitting form. When no input values, load a blank form. 
    previous_form = None
    if previous_form is None: previous_form = BudgetForm()

    # User post request when fill out form of rent budget/property features
    if request.method == 'POST':
        budget_form = BudgetForm(request.POST)
        # If valid input, update budget, features and retain inputs as previous_form
        if budget_form.is_valid(): 
            budget, features = get_sample_data(budget_form)
            previous_form = budget_form
        # If invalid input on form, render user with same form
        else: 
            return render(request, "compare_ml/log_model.html", {
                'plot': plot, 
                'form': budget_form,
            })
        # Overwrite plot using sample rent budget/property features input in form
        plot, prediction, prediction_prob, accuracy, f1 = log_model(budget, features)
    
    # Render plot (with either default features or overwritten by sample features)
    return render(request, "compare_ml/log_model.html", {
        'plot': plot, 
        'form': previous_form,
    })


# K-NEAREST NEIGHBOURS ML MODEL VIEW
###### about 4 second runtime
def knn_model_plot(request):
    # Get plot using default rent budget/property features
    plot, prediction, prediction_prob, accuracy, f1 = knn_model(5000, [940, 5, 1, 0, 0])
    # plot, prediction, prediction_prob, accuracy, f1 = knn_model(5000, [940, 5, 1, 0, 0, 0, 0, 0, 0])

    # Maintain input values upon submitting form. When no input values, load a blank form. 
    previous_form = None
    if previous_form is None: previous_form = BudgetForm()

    # User post request when fill out form of rent budget/property features
    if request.method == 'POST':
        budget_form = BudgetForm(request.POST)
        # If valid input, update budget, features and retain inputs as previous_form
        if budget_form.is_valid(): 
            budget, features = get_sample_data(budget_form)
            previous_form = budget_form
        # If invalid input on form, render user with same form
        else: 
            return render(request, "compare_ml/knn_model.html", {
                'plot': plot, 
                'form': budget_form,
            })
        # Overwrite plot using sample rent budget/property features input in form
        plot, prediction, prediction_prob, accuracy, f1 = knn_model(budget, features)
        print('~~~~~FEATURES~~~~', features)

    return render(request, "compare_ml/knn_model.html", {
        'plot': plot,
        'form': previous_form,
    }) 



class BudgetForm(forms.Form):
    # choice options for time to subway and bedroom numbers for property
    time_to_subway = (
        ('1', '<5 mins'),
        ('2', '5-15 mins'),
        ('3', '15-30 mins'),
        ('4', '30+ mins'),
    )
    bedroom_nums = (
        ('1', '1 bedrooms'),
        ('2', '2 bedrooms'),
        ('3', '3 bedrooms'),
        ('4', '4 bedrooms'),
        ('5', '5 bedrooms'),
    )

    # Form input fields for rent budget and propety features
    budget = forms.IntegerField(label="Monthly?? Rent Budget (USD)", min_value=2000, max_value=20000)
    size = forms.IntegerField(label="Size (sqft)", min_value=0, max_value=5000)
    subway = forms.ChoiceField(choices = time_to_subway, label="subway")
    bedrooms = forms.ChoiceField(choices = bedroom_nums, label="bedrooms")
    # one_bed = forms.BooleanField(label="one_bed", required=False)
    gym = forms.BooleanField(label="gym", required=False)
    patio = forms.BooleanField(label="patio", required=False)


def get_sample_data(budget_form):
    # Labels
    budget = budget_form.cleaned_data['budget']

    # Features
    size = budget_form.cleaned_data['size']

    if budget_form.cleaned_data['subway'] == '1': mins_to_subway = 2.5
    elif budget_form.cleaned_data['subway'] == '2': mins_to_subway = 10
    elif budget_form.cleaned_data['subway'] == '3': mins_to_subway = 22.5
    else: mins_to_subway = 45

    # one_bed, two_bed, three_bed, four_bed, five_bed = 0, 0, 0, 0, 0
    # rooms = lambda x: 1 if f'{x}' == budget_form.cleaned_data['bedrooms'] else 0
    # one_bed, two_bed, three_bed, four_bed, five_bed = rooms(1), rooms(2), rooms(3), rooms(4), rooms(5)

    bedrooms = int(budget_form.cleaned_data['bedrooms'])

    bool = lambda x: 1 if x is True else 0
    # one_bed = bool(budget_form.cleaned_data['one_bed'])
    gym = bool(budget_form.cleaned_data['gym'])
    patio = bool(budget_form.cleaned_data['patio'])
    features = [size, mins_to_subway, bedrooms, gym, patio]
    # features = [size, mins_to_subway, one_bed, two_bed,three_bed,four_bed, five_bed, gym, patio]
    
    # print('~~~~FEATURES~~~~~', features)
    # element = features[2]
    # print(element)
    # print(type(element))
    # ~~~~FEATURES~~~~~ [1, 2.5, 1, 0, 0] ONE_BED, class int
    # ~~~~FEATURES~~~~~ [1, 2.5, '1', 0, 0] ONE_BED, class str

    return budget, features
