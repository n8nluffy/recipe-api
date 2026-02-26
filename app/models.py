from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class Recipe(models.Model):
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    time_minutes = models.IntegerField()
    price = models.DecimalField(max_digits=5, decimal_places=2)

    def __str__(self):
        return self.title


class RecipeRating(models.Model):
    recipe = models.ForeignKey(Recipe, on_delete=models.CASCADE, related_name="ratings")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="recipe_ratings"
    )
    stars = models.PositiveSmallIntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)]
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("recipe", "user")

    def __str__(self):
        return f"{self.user} rated {self.recipe} with {self.stars} stars"
