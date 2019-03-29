from django.db import models

# Create your models here.

class Author(models.Model):
      name = models.CharField(max_length=255)
      email = models.EmailField()


class Article(models.Model):
    title = models.CharField(max_length=120)
    description = models.TextField()
    body = models.TextField()
    # author = models.ForeignKey('Author', related_name='articles')