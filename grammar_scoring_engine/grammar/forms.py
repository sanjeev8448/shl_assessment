from django import forms
from .models import AudioSample

class AudioForm(forms.ModelForm):
    class Meta:
        model = AudioSample
        fields = ['audio_file']
