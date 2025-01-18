import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import pickle

# Load the cleaned data
with open('cleaned_data.pkl', 'rb') as file:
    df = pickle.load(file)


# Let's first look at what we have in our cleaned dataset
print("Dataset Overview:")
print(f"Number of recipes: {len(df)}")
print("\nColumns in our dataset:")
for column in df.columns:
    print(f"- {column}")

class RecipeRecommender:
    def __init__(self, recipe_df):
        """
        Initialize the recipe recommendation system with our cleaned dataset.
        
        Parameters:
        recipe_df (pandas.DataFrame): Our cleaned recipe dataset containing:
            - ingredients: List of ingredients for each recipe
            - estimated_cooking_time: Cooking duration in minutes
            - number_of_steps: Number of steps in the recipe
            - directions: Cooking instructions
        """
        self.df = recipe_df
        # Initialize TF-IDF vectorizer for ingredient analysis
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Create ingredient feature matrix
        # Convert ingredient lists to strings for TF-IDF processing
        ingredient_texts = self.df['ingredients'].apply(
            lambda x: ' '.join(ast.literal_eval(x)) if isinstance(x, str) else ''
        )
        self.ingredients_matrix = self.vectorizer.fit_transform(ingredient_texts)

    def recommend_recipes(self, ingredients, max_time=None, max_steps=None, n_recommendations=5):
        """
        Find and recommend recipes based on available ingredients and constraints.
        
        Parameters:
        ingredients (list): List of ingredients available to the user
        max_time (int, optional): Maximum cooking time in minutes
        max_steps (int, optional): Maximum number of steps preferred
        n_recommendations (int): Number of recipes to recommend
        
        Returns:
        pandas.DataFrame: Recommended recipes with similarity scores
        """
        # Convert user ingredients to TF-IDF format
        user_ingredients = ' '.join(ingredients)
        user_vector = self.vectorizer.transform([user_ingredients])
        
        # Calculate similarity scores with all recipes
        similarities = cosine_similarity(user_vector, self.ingredients_matrix)[0]
        
        # Create a mask for filtering recipes
        mask = np.ones(len(self.df), dtype=bool)
        
        # Apply time constraint if specified
        if max_time is not None:
            mask &= self.df['estimated_cooking_time'] <= max_time
            
        # Apply steps constraint if specified
        if max_steps is not None:
            mask &= self.df['number_of_steps'] <= max_steps
        
        # Get final recommendations
        valid_indices = np.where(mask)[0]
        valid_similarities = similarities[mask]
        
        # Sort by similarity and get top recommendations
        top_indices = valid_indices[np.argsort(valid_similarities)[-n_recommendations:]]
        
        # Create results dataframe
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        # Sort by similarity score
        return recommendations.sort_values('similarity_score', ascending=False)

# Create a user-friendly interface
def get_user_preferences():
    """
    Collect user preferences through an interactive prompt.
    """
    print("🍳 Welcome to the Recipe Recommender! 🍳")
    
    # Get ingredients
    print("\nWhat ingredients do you have? (separate with commas)")
    print("Example: chicken, rice, onion, garlic")
    ingredients = input("> ").strip().split(',')
    ingredients = [ing.strip().lower() for ing in ingredients]
    
    # Get time constraint
    print("\nWhat's your maximum cooking time in minutes?")
    print("(Press Enter to skip this constraint)")
    time_input = input("> ").strip()
    max_time = int(time_input) if time_input.isdigit() else None
    
    # Get step constraint
    print("\nMaximum number of steps you're comfortable with?")
    print("(Press Enter to skip this constraint)")
    steps_input = input("> ").strip()
    max_steps = int(steps_input) if steps_input.isdigit() else None
    
    return {
        'ingredients': ingredients,
        'max_time': max_time,
        'max_steps': max_steps
    }

def display_recipe_recommendations(recommendations):
    """
    Display recommended recipes in a clear, formatted way.
    """
    print("\n📋 Your Recommended Recipes 📋")
    print("=" * 50)
    
    for idx, recipe in recommendations.iterrows():
        print(f"\n🔸 Recipe: {recipe['title']}")
        print(f"  Similarity Score: {recipe['similarity_score']:.2f}")
        print(f"  Cooking Time: {recipe['estimated_cooking_time']} minutes")
        print(f"  Number of Steps: {recipe['number_of_steps']}")
        
        # Display ingredients
        print("\n  Ingredients needed:")
        ingredients = ast.literal_eval(recipe['ingredients'])
        for ingredient in ingredients:
            print(f"   • {ingredient}")
            
        print("-" * 50)

# Main execution function
def main():
    # Initialize recommender with our cleaned data
    recommender = RecipeRecommender(df)
    
    while True:
        # Get user preferences
        preferences = get_user_preferences()
        
        # Get recommendations
        recommendations = recommender.recommend_recipes(
            ingredients=preferences['ingredients'],
            max_time=preferences['max_time'],
            max_steps=preferences['max_steps']
        )
        
        # Display recommendations
        display_recipe_recommendations(recommendations)
        
        # Ask if user wants to try again
        print("\nWould you like to try another search? (yes/no)")
        if input("> ").lower().strip() != 'yes':
            break
    
    print("\nThank you for using the Recipe Recommender! Happy cooking! 👨‍🍳")

if __name__ == "__main__":
    main()