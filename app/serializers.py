from rest_framework import serializers

from .models import Recipe, RecipeRating


class RecipeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Recipe
        fields = ["id", "title", "description", "time_minutes", "price"]


class RecipeRatingSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecipeRating
        fields = ["id", "recipe", "user", "stars", "created_at"]
        read_only_fields = ["created_at"]
