from django.shortcuts import render
from users.forms import UserRegistrationForm
from django.http import FileResponse
import os


# Create your views here.
def index(request):
    return render(request, 'index.html', {})


def logout(request):
    return render(request, 'index.html', {})


def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})


def learn_more(request):
    filepath = os.path.join('static/pdf', 'rainfall.pdf')
    return FileResponse(open(filepath, 'rb'), content_type='application/pdf')
