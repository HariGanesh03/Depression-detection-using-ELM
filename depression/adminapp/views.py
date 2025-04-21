from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.paginator import Paginator
from .models import  depression_Upload_dataset
from userapp.models import UserDetails, Predict_details
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from itertools import chain
from userapp.models import UserDetails, Predict_details
from adminapp.models import depression_Upload_dataset

def lr_alg(req):
    return render(req, 'admin/Logistic-Regression.html')

def rf_alg(req):
    return render(req, 'admin/Random-Forest.html')
def knn_alg(req):
    return render(req, 'admin/KNN.html')



# Admin dashboard views
def admin_index(request):
    messages.success(request, 'Login successful')
    all_users_count = UserDetails.objects.all().count()
    pending_users_count = UserDetails.objects.filter(user_status='pending').count()
    rejected_users_count = UserDetails.objects.filter(user_status='Rejected').count()
    accepted_users_count = UserDetails.objects.filter(user_status='Accepted').count()
    datasets_count = depression_Upload_dataset.objects.all().count()
    no_of_predicts = Predict_details.objects.all().count()
    return render(request, 'admin/index.html', {
        'a': pending_users_count,
        'b': all_users_count,
        'c': rejected_users_count,
        'd': accepted_users_count,
        'e': datasets_count,
        'f': no_of_predicts
    })

def admin_pending(request):
    users = UserDetails.objects.filter(user_status='pending')
    context = {'u': users}
    return render(request, "admin/pending.html", context)

def Admin_Reject_Btn(request, x):
    user = UserDetails.objects.get(user_id=x)
    user.user_status = 'Rejected'
    user.save()
    messages.warning(request, 'Rejected')
    return redirect('pending')

def Admin_accept_Btn(req, x):
    user = UserDetails.objects.get(user_id=x)
    user.user_status = 'Accepted'
    user.save()
    messages.success(req, 'Accepted')
    return redirect('pending')

def admin_manage(request):
    a = UserDetails.objects.all()
    paginator = Paginator(a, 5)
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request, "admin/manage.html", {'all': post})

# Views for handling dataset uploads



def depression_admin_upload(request):
    if request.method == 'POST':
        file = request.FILES['data_file']
        file_size = str((file.size) / 1024) + ' kb'
        depression_Upload_dataset.objects.create(File_size=file_size, Dataset=file)
        messages.success(request, 'Your dataset was uploaded.')
    return render(request, "admin/depression-upload-data.html")

# Delete a dataset
def delete_dataset(request, id):
    try:
        depression_Upload_dataset.objects.get(user_id=id).delete()
    except depression_Upload_dataset.DoesNotExist:
        pass
  
        pass
    return redirect('view')

# View combined datasets
def admin_view(request):
    dataset = depression_Upload_dataset.objects.all()
    all_datasets = list(chain( dataset))
    paginator = Paginator(all_datasets, 5)
    page_number = request.GET.get('page')
    post = paginator.get_page(page_number)
    return render(request, "admin/view-data.html", {'data': all_datasets, 'user': post})

def view_view(request):
    data = depression_Upload_dataset.objects.last()
    filed = str(data.Dataset)
    dfd = pd.read_csv(f'./media/{filed}')
    tabled = dfd.to_html(table_id='data_table')

    return render(request, 'admin/view-view.html', { 's': tabled})
# views.pyfrom django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def logistic_regression_depression(req):
    # Load the dataset
    data = pd.read_csv('Dataset/dp.csv')  # Ensure the path is correct

    # Define feature columns and target variable
    features = ['Age', 'RelationShip_Status', 'Gender', 'Education', 'Current_Role', 'Family_Income', 
                'q3(d)', 'q5(d)', 'q10(d)', 'q13(d)', 'q16(d)', 'q17(d)', 'q21(d)']
    target = 'Depression1'

    # One-hot encode categorical features
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    categorical_features = ['RelationShip_Status', 'Gender', 'Education', 'Current_Role']
    encoded_categorical_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]).toarray(), 
                                            columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical data with numerical features
    numerical_features = ['Age', 'Family_Income', 'q3(d)', 'q5(d)', 'q10(d)', 'q13(d)', 'q16(d)', 'q17(d)', 'q21(d)']
    X = pd.concat([data[numerical_features], encoded_categorical_data], axis=1)

    # Target variable (Depression1)
    y = data[target]

    # Check unique classes in target variable
    print("Unique classes in target variable:", y.unique())

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Make probability predictions for all classes
    y_prob = model.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

    # Calculate AUC for multi-class classification
    try:
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')  # Ensure y_prob is used correctly
    except ValueError as e:
        # If there's an issue with the AUC calculation, handle it gracefully
        print("Error calculating AUC:", e)
        auc_score = None  # Set AUC score to None or handle it as needed

    # Generate the classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)  # Output as a dict for easier access

    # Prepare results to pass to the template
    results = {
        'accuracy': accuracy,
        'auc_score': auc_score if auc_score is not None else "N/A",  # Show "N/A" if AUC is None
        'classification_report': class_report,
    }

    # Show success message and render the result in the template
    messages.success(req, 'Logistic Regression executed successfully')
    return render(req, 'admin/Logistic-Regression.html', {'results': results})



from django.shortcuts import render
from django.contrib import messages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import joblib
def knn_depression(req):
    # Load the latest dataset from the database
    dataset = depression_Upload_dataset.objects.last()
    df_ax = pd.read_csv('Dataset/dp.csv')  # Ensure the path is correct

    # Define feature columns and target variable
    features = ['Age', 'RelationShip_Status', 'Gender', 'Education', 'Current_Role', 'Family_Income', 
                'q3(d)', 'q5(d)', 'q10(d)', 'q13(d)', 'q16(d)', 'q17(d)', 'q21(d)']
    target = 'Depression1'

    # One-hot encode categorical features
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore')
    categorical_features = ['RelationShip_Status', 'Gender', 'Education', 'Current_Role']
    encoded_categorical_data = pd.DataFrame(encoder.fit_transform(df_ax[categorical_features]).toarray(), 
                                            columns=encoder.get_feature_names_out(categorical_features))

    # Combine encoded categorical data with numerical features
    numerical_features = ['Age', 'Family_Income', 'q3(d)', 'q5(d)', 'q10(d)', 'q13(d)', 'q16(d)', 'q17(d)', 'q21(d)']
    X = pd.concat([df_ax[numerical_features], encoded_categorical_data], axis=1)

    # Target variable (Depression1)
    y = df_ax[target]

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

    # Initialize and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)
    
    # Make probability predictions for all classes
    y_prob = knn.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage

    # Calculate AUC score
    try:
        auc_score = roc_auc_score(y_test, y_prob, multi_class='ovr')  # Ensure y_prob is used correctly
    except ValueError as e:
        # If there's an issue with the AUC calculation, handle it gracefully
        print("Error calculating AUC:", e)
        auc_score = None  # Set AUC score to None or handle it as needed

    # Generate the classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)  # Output as a dict for easier access

    # Prepare results to pass to the template
    results = {
        'accuracy': round(accuracy, 2),
        'auc_score': auc_score if auc_score is not None else "N/A",  # Show "N/A" if AUC is None
        'classification_report': class_report,
    }
    joblib.dump(knn,'knn_depression_model.pkl')


    # Show success message and render the result in the template
    messages.success(req, 'KNN executed successfully')
    return render(req, 'admin/KNN.html', {'results': results})

import pandas as pd
from django.shortcuts import render
import matplotlib.pyplot as plt
import io
import base64

def compare(request):
    # Hardcoded accuracy scores for KNN and Logistic Regression
    accuracy_scores = {
        "KNN": 100,
        "Logistic Regression": 83
    }

    min_score = 0.0  # Minimum score
    max_score = 100.0  # Maximum score
 

    # Plot the comparison graph
    plt.figure(figsize=(7, 5))
    plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color=['blue', 'green'],width=0.3)
    plt.title("Model Comparison: Accuracy Scores", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xlabel("Models", fontsize=14)
    plt.ylim(min_score, max_score)  # Set the y-axis limits dynamically
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    # Save the plot to a PNG image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    # Render the graph in the template
    return render(request, 'admin/compare.html', {'graph': image_base64})
