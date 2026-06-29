from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User

from .models import Candidate, Election


class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Enter username'})
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Enter password'})
    )


class ElectionForm(forms.ModelForm):
    starts_at = forms.DateTimeField(
        input_formats=['%Y-%m-%dT%H:%M'],
        widget=forms.DateTimeInput(attrs={'type': 'datetime-local'}),
    )
    ends_at = forms.DateTimeField(
        input_formats=['%Y-%m-%dT%H:%M'],
        widget=forms.DateTimeInput(attrs={'type': 'datetime-local'}),
    )

    class Meta:
        model = Election
        fields = ('name', 'description', 'starts_at', 'ends_at', 'status')
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'Election name'}),
            'description': forms.Textarea(attrs={'rows': 4, 'placeholder': 'Election description'}),
            'status': forms.Select(),
        }


class CandidateForm(forms.ModelForm):
    class Meta:
        model = Candidate
        fields = ('election', 'name', 'party', 'manifesto', 'display_order')
        widgets = {
            'election': forms.Select(),
            'name': forms.TextInput(attrs={'placeholder': 'Candidate name'}),
            'party': forms.TextInput(attrs={'placeholder': 'Party or group'}),
            'manifesto': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Short manifesto or notes'}),
            'display_order': forms.NumberInput(attrs={'min': 0, 'value': 0}),
        }
