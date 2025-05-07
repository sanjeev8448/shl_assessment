from django.db import models

class AudioSample(models.Model):
    audio_file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Audio Sample: {self.audio_file.name}"

class GrammarScore(models.Model):
    audio_sample = models.ForeignKey(AudioSample, on_delete=models.CASCADE)
    transcribed_text = models.TextField()
    score = models.FloatField()

    def __str__(self):
        return f"Score for {self.audio_sample.audio_file.name}: {self.score}/10"
