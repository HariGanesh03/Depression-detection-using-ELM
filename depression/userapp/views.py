from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.mail import send_mail
from django.conf import settings
from userapp.models import *
import urllib.request
import pandas as pd
import time
from adminapp.models import *
import urllib.parse
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np


# Create your views here.
def sendSMS(user, otp, mobile):
    data = urllib.parse.urlencode({
        'username': 'Codebook',
        'apikey': '56dbbdc9cea86b276f6c',
        'mobile': mobile,
        'message': f'Hello {user}, your OTP for account activation is {otp}. This message is generated from https://www.codebook.in server. Thank you',
        'senderid': 'CODEBK'
    })
    data = data.encode('utf-8')
    # Disable SSL certificate verification
    # context = ssl._create_unverified_context()
    request = urllib.request.Request("https://smslogin.co/v3/api.php?")
    f = urllib.request.urlopen(request, data)
    return f.read()

def user_services(request):
    return render(request, 'user/services.html')

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserDetails

def user_register(request):
    if request.method == 'POST':
        username = request.POST.get('user_name')
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')
        number = request.POST.get('contact_number')
        file = request.FILES['user_file']

        # Check if the email is already registered
        if UserDetails.objects.filter(user_email=email).exists():
            messages.info(request, 'Email already registered.')
            return redirect('register')

        # Save user details in the database
        UserDetails.objects.create(
            user_username=username,
            user_email=email,
            user_password=password,
            user_contact=number,
            user_file=file
        )
        messages.success(request, 'Registration successful!')
        return redirect('login')  # Redirect to login page

    return render(request, 'user/register.html')

from django.shortcuts import render, redirect
from django.contrib import messages
from .models import UserDetails

def user_login(request):
    if request.method == 'POST':
        email = request.POST.get('email_address')
        password = request.POST.get('email_password')

        try:
            # Check if user exists with provided email and password
            user = UserDetails.objects.get(user_email=email, user_password=password)

            # Validate account status
            if user.user_status == 'Accepted':
                request.session['user_id'] = user.user_id
                user.No_Of_Times_Login += 1  # Increment login count
                user.save()
                messages.success(request, 'Login successful!')
                return redirect('dashboard')
            elif user.user_status == 'Rejected':
                messages.warning(request, "Your account has been rejected.")
            elif user.user_status == 'Pending':
                messages.info(request, "Your account is pending approval.")
        except UserDetails.DoesNotExist:
            # Handle invalid credentials
            messages.error(request, 'Invalid login credentials. Please try again.')
            return redirect('login')

    return render(request, "user/user.html")

def user_admin(request):
    admin_name = 'admin@gmail.com'
    admin_password = 'admin'
    if request.method == 'POST':
        adminemail = request.POST.get('emailaddress')
        adminpassword = request.POST.get('emailpassword')
        if admin_name == adminemail and admin_password == adminpassword:
            messages.success(request,'login successfull')

            return redirect('admin_dashboard')
        
        else:
            messages.error(request,"login credentials was incorrect....")

            return redirect('admin')
    return render(request, "user/admin.html")

def user_otp(request):
    user_id = request.session['user_email']
    user =UserDetails.objects.get(user_email = user_id)
    messages.success(request, 'OTP  Sent successfully')
    print(user_id)
    print(user, 'user avilable')
    print(type(user.otp))
    print(user. otp, 'creaetd otp')   
    if request.method == 'POST':
        u_otp = request.POST.get('otp')
        u_otp = int(u_otp)
        print(u_otp, 'enter otp')
        if u_otp == user.otp:
            print('if')
            user.otp_status  = 'verified'
            user.save()
            messages.success(request, 'OTP  verified successfully')
            return redirect('login')
        else:
            print('else')
            messages.error(request, 'Invalid OTP  ') 
            return redirect('otp')
    return render(request, 'user/otp.html')

def user_index(request):
    return render(request, 'user/index.html')
 
def user_about(request):
    return render(request, "user/about.html")

def user_contact(request):
    return render(request, "user/contact.html")

def user_dashboard(request):
    prediction_count =  UserDetails.objects.all().count()
    user_id = request.session["user_id"]
    user = UserDetails.objects.get(user_id = user_id)
    return render(request, "user/dashboard.html", {'predictions' : prediction_count, 'la' : user})

def user_myprofile(request):
    views_id = request.session['user_id']
    user = UserDetails.objects.get(user_id = views_id)
    print(user, 'user_id')
    if request.method =='POST':
        username = request.POST.get('user_name')
        email = request.POST.get('email_address')
        number = request.POST.get('contact_number')
        password = request.POST.get('email_password')
        age = request.POST.get('Age_int')
        date = request.POST.get('date')
        print(username, email, number, password, date, age, 'data') 

        user.user_username = username
        user.user_email = email
        user.user_contact = number
        user.user_password = password
        user.user_dates = date 

        if len(request.FILES)!=0:
            file = request.FILES['user_file']
            user.user_file = file
            user.user_username = username
            user.user_email = email
            user.user_contact = number
            user.user_password = password
            user.save()
            messages.success(request, 'Updated Successfully...!')

        else:
            user.user_username = username
            user.user_email = email
            user.user_contact = number
            user.user_password = password
            user.save()
            messages.success(request, 'Updated Successfully...!')


    return render(request, "user/myprofile.html", {'i':user})





from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from django.shortcuts import redirect, render
from django.contrib import messages
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

def user_depression(req):
    if req.method == 'POST':
        # Get form data and convert to integers
        age = int(req.POST.get('age'))
        RelationShip = int(req.POST.get('RelationShip'))
        Family = int(req.POST.get('Family'))
        Current = int(req.POST.get('Current'))
        Education = int(req.POST.get('Education'))
        Gender = int(req.POST.get('Gender'))
        q1_nutrition = int(req.POST.get('q1_nutrition'))
        q2_screentime = int(req.POST.get('q2_screentime'))
        q3_screentime = int(req.POST.get('q3_screentime'))
        q4_screentime = int(req.POST.get('q4_screentime'))
        q5_screentime = int(req.POST.get('q5_screentime'))
        q6_screentime = int(req.POST.get('q6_screentime'))
        q7_frequency = int(req.POST.get('q7_frequency'))

        # Load the pre-trained KNN model
        file_path = 'C:/Users/91910/Desktop/depression/knn_model.pkl'  # Path to the saved model file
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)

        # Make a prediction with user inputs
        res = loaded_model.predict([[age, RelationShip, Gender, Education, Current, Family, 
                                      q1_nutrition, q2_screentime, q3_screentime, 
                                      q4_screentime, q5_screentime, q6_screentime, q7_frequency]])
        result_int = int(res[0])
        
        # Load dataset for evaluation
        df = pd.read_csv('Dataset/dp.csv')
        X = df.drop('Depression1', axis=1)
        y = df['Depression1']

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

        # Initialize and fit the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(X_train, y_train)

        # Make predictions
        y_pred = knn.predict(X_test)

        # Make probability predictions for AUC calculation
        y_prob = knn.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

        # Calculate AUC score
        try:
            auc_score = roc_auc_score(y_test, y_prob)  # Ensure proper calculation
        except ValueError as e:
            print("Error calculating AUC:", e)
            auc_score = None  # Set AUC score to None if calculation fails

        # Store metrics in session
        req.session['acrd'] = round(accuracy, 2)
        req.session['auc'] = round(auc_score, 2) if auc_score is not None else "N/A"

        # Set result messages based on prediction
        if result_int >= 0 and result_int <= 8:
            messages.success(req, "Normal")
        elif result_int >= 10 and result_int <= 12:
            messages.warning(req, "Mild")
        elif result_int >= 14 and result_int <= 20:
            messages.warning(req, "Moderate")
        elif result_int >= 22 and result_int <= 26:
            messages.warning(req, "Severe")
        else:
            messages.warning(req, "Extremely Severe")

        return redirect("depression_result")
    
    return render(req, 'user/depression.html')





def userlogout(request):
    view_id = request.session["user_id"]
    user = UserDetails.objects.get(user_id = view_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(request, 'You are logged out..')
    # print(user.Last_Login_Time)
    # print(user.Last_Login_Date)
    return redirect('login')


def user_feedback(request):
    return render(request, 'user/feedback.html')

def user_result_anxiety(request):
    accuracy = request.session.get('acra')
    precession = request.session.get('prea')
    recall = request.session.get('reca')
    f1 = request.session.get('fa')
    x = request.session.get('resa')

    return render(request, "user/anxiety-result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})


def user_result_Depression(request):
    accuracy = request.session.get('acrd')
    precession = request.session.get('pred')
    recall = request.session.get('recd')
    f1 = request.session.get('fd')
    x = request.session.get('resd')

    return render(request, "user/Depression-result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})


def user_result_Stress(request):
    accuracy = request.session.get('acrs')
    precession = request.session.get('pres')
    recall = request.session.get('recs')
    f1 = request.session.get('fs')
    x = request.session.get('ress')
    print(accuracy,precession,recall,f1,x)

    return render(request, "user/stress-result.html",{'accuracy':accuracy, 'precession':precession, 'recall': recall, 'f1':f1, 'res':x})
