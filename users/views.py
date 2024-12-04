from django.conf import settings
from django.shortcuts import render
# from django.http import HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import pandas as pd


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have  successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Exists')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('password')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by Admin.')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def ml(request):
    if request.method == 'POST':
        temp_high = int(request.POST.get('temp_high'))
        temp_avg = int(request.POST.get('temp_avg'))
        temp_low = int(request.POST.get('temp_low'))
        dew_point_high = int(request.POST.get('dew_point_high'))
        dew_point_avg = int(request.POST.get('dew_point_avg'))
        dew_point_low = int(request.POST.get('dew_point_low'))
        humidity_high = int(request.POST.get('humidity_high'))
        humidity_avg = int(request.POST.get('humidity_avg'))
        humidity_low = int(request.POST.get('humidity_low'))
        sea_level_pressure_avg_inches = float(request.POST.get('sea_level_pressure_avg_inches'))
        visibility_high = int(request.POST.get('visibility_high'))
        visibility_avg = int(request.POST.get('visibility_avg'))
        visibility_low = int(request.POST.get('visibility_low'))
        wind_high = int(request.POST.get('wind_high'))
        wind_avg = int(request.POST.get('wind_avg'))
        wind_gust = int(request.POST.get('wind_gust'))

        # print(temp_high, temp_low, temp_avg, dew_point_high, dew_point_low, dew_point_avg,
        #       humidity_low, humidity_avg, humidity_high, sea_level_pressure_avg_inches,
        #       visibility_avg, visibility_low, visibility_high, wind_avg, wind_high, wind_gust, sep='\n')
        #
        from .utility.ml import do_prediction
        result = do_prediction(
            temp_high=temp_high,
            temp_avg=temp_avg,
            temp_low=temp_low,

            dew_point_high=dew_point_high,
            dew_point_avg=dew_point_avg,
            dew_point_low=dew_point_low,

            humidity_high=humidity_high,
            humidity_avg=humidity_avg,
            humidity_low=humidity_low,

            sea_level_pressure_avg_inches=sea_level_pressure_avg_inches,
            visibility_high=visibility_high,
            visibility_avg=visibility_avg,
            visibility_low=visibility_low,

            wind_high=wind_high,
            wind_avg=wind_avg,
            wind_gust=wind_gust,

        )
        return render(request, 'users/ml.html', {'result': f"The precipitation in inches for the input is:{result}"})
        # return render(request, 'users/prediction.html')


    else:
        # return render(request, 'users/prediction.html')
        return render(request, 'users/ml.html')


def dataset(request):
    dataset_url = settings.DATASET_URL
    data = pd.read_csv(dataset_url)
    context = {
        'data': data.to_html(
            index=False,
            classes=['table table-striped table-bordered table-hover table-sm']
        ).replace('<tr style="text-align: right;">', '<tr>')
    }

    return render(request, 'users/view_data.html', context)


def ann(request):
    from .utility import artificial_neural_network as ann
    print('Lets print the results... ')
    regressor = ann.build_regressor()
    # Evaluate Loss (Mean Squared Error), Mean Absolute Error, Accuracy,
    regressor_results = regressor.evaluate(ann.X_test, ann.y_test)
    print("*************** Regressor Result ***************")

    loss = regressor_results[0]
    mae = regressor_results[1]
    accuracy = regressor_results[2]
    print('__LOSS__:', loss)
    print('__MAE__:', mae)
    print('__ACCURACY__:', accuracy)

    context = {
        'loss': loss,
        'mae': mae,
        'accuracy': accuracy
    }
    return render(request, 'users/ann.html', context)


def mlr(request):
    from .utility.ml import mae_mse_r2_score
    result = mae_mse_r2_score()

    context = {
        'mean_absolute_error': result[0],
        'mean_squared_error': result[1],
        'r2_score': result[2]
    }

    return render(request, 'users/mlr.html', context)
