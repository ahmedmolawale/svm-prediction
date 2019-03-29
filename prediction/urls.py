from django.urls import path

from .views import DiagnoseView
from .views import TreatmentView

app_name = "predictions"

# app_name will help us do a reverse look-up latter.
urlpatterns = [
    path('diagnose', DiagnoseView.as_view()),
    path('treatment', TreatmentView.as_view())
]