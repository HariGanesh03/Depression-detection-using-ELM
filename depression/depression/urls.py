"""
URL configuration for hearing_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from userapp import views as user_views
from adminapp import views as admin_view
# URLS
urlpatterns = [
    # User_Urls
    path('admin/', admin.site.urls),
    path('user-register/', user_views.user_register, name = 'register'),
    path('user-login/', user_views.user_login, name = 'login'),
    path('admin-login/', user_views.user_admin, name = 'admin'),
    path('user-otp/', user_views.user_otp, name = 'otp'),
    path('', user_views.user_index, name = "index"),
    path('user-dashboard/', user_views.user_dashboard, name = 'dashboard'),
    path('user-aboutus/', user_views.user_about, name = "about"),
    path('user-contact/', user_views.user_contact, name = "contact"),
    path('user-myprofile/', user_views.user_myprofile, name = 'myprofile'),


    path('user-depression-result/', user_views.user_result_Depression, name ="depression_result"),
    
    path('userlogout/', user_views.userlogout, name = 'userlogout'),
    path('user-feedback/', user_views.user_feedback, name = 'feedback'),
    path('user-depression/', user_views.user_depression, name = 'depression'),
    


    #URLS_admin
    path('admin-dashboard/', admin_view.admin_index, name = "admin_dashboard"),
    path('admin-pendingusers/', admin_view.admin_pending, name = "pending"),
    path('admin-manageusers/', admin_view.admin_manage, name = 'manage'),
    # path('admin-anxiety-upload/', admin_view.admin_upload, name = 'upload'),
    # path('admin-stress-upload/', admin_view.stress_admin_upload, name = 'stress_upload'),
    path('admin-depression-upload/', admin_view.depression_admin_upload, name = 'depression_upload'),
    path('admin-viewdata/', admin_view.admin_view, name = "view"),

    path('linear-regression/', admin_view.lr_alg, name='linear_regression'),
    path('random-forest/', admin_view.rf_alg, name='random_forest'),
    path('knn/', admin_view.knn_alg, name='knn'),


    path('logistic_regression_depression/', admin_view.logistic_regression_depression, name='logistic_regression_depression'),
    path('knn-depression/', admin_view.knn_depression, name='knn_depression'),
    path('compare/', admin_view.compare, name = 'compare'),



    # path('admin-comparison-grapha/', admin_view.admin_comparison_graph_a, name = "comparison-graph-a"),
    # path('admin-comparison-graphs/', admin_view.admin_comparison_graph_s, name = "comparison-graph-s"),
    # path('admin-comparison-graphd/', admin_view.admin_comparison_graph_d, name = "comparison-graph-d"),
    path('adminrejectbtn/<int:x>', admin_view.Admin_Reject_Btn, name='adminreject'),
    path('adminacceptbtn/<int:x>', admin_view.Admin_accept_Btn, name='adminaccept'),
    # path('admin-change-status/<int:id>',admin_view.Change_Status, name ='change_status'),
    # path('admin-delete/<int:id>',admin_view.Delete_User, name ='delete_user'), 
    path('delete-dataset/<int:id>', admin_view.delete_dataset, name = 'delete_dataset'),
    path('view_view/', admin_view.view_view, name='view_view'),

]   + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
