from django.shortcuts import render
from .forms import AudioForm
from .models import AudioSample, GrammarScore
from .ai_engine import transcribe, predict_score

def upload_audio(request):
    score = None
    text = ""
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio_sample = form.save()
            path = audio_sample.audio_file.path
            text = transcribe(path)
            score = predict_score(text)
            GrammarScore.objects.create(audio_sample=audio_sample, transcribed_text=text, score=score)
    else:
        form = AudioForm()
    return render(request, 'upload.html', {'form': form, 'text': text, 'score': score})
