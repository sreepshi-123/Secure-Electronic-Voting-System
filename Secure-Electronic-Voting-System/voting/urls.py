from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),
    path('home/', views.home, name='home'),
    path('superuser-panel/', views.superuser_panel, name='superuser_panel'),
    path('register/', views.register_voter, name='register'),
    path('vote/', views.vote, name='vote'),
    path('results/', views.results, name='results'),
]
