from django.db import models

class depression_Upload_dataset(models.Model):
    # Add your fields for the uploaded dataset here
    Dataset = models.FileField(upload_to='datasets/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    File_size = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f'Dataset uploaded on {self.uploaded_at}'

class RandomForest(models.Model):
    # This model will store the results from various algorithms
    Accuracy = models.FloatField()
    AUC = models.FloatField()
    URL = models.URLField()
    Name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.Name} - Accuracy: {self.Accuracy}, AUC: {self.AUC}"





