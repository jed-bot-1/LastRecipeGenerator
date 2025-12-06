from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Recipe Recommendation API",
    description="API for recommending recipes based on ingredient similarity",
    version="3.0.0"
)

# Global variables
recipes_df = None
all_ingredients = None
recipe_vectors = None  # Will store TF-IDF or binary vectors

class RecipeRequest(BaseModel):
    """Request model for recipe recommendations"""
    ingredients: List[str]

class RecipeResponse(BaseModel):
    """Response model for recipe recommendations"""
    status: str
    message: str
    timestamp: str
    input_ingredients: List[str]
    recommendations: List[dict]

def load_and_prepare_data():
    """Load recipes and create ingredient vectors"""
    global recipes_df, all_ingredients, recipe_vectors
    
    try:
        # Load recipes
        recipes_df = pd.read_csv('recipes_processed.csv')
        print(f"✅ Loaded {len(recipes_df)} recipes")
        
        # Extract all unique ingredients
        all_ingredients = set()
        for idx, row in recipes_df.iterrows():
            if 'ingredient_set' in row and pd.notna(row['ingredient_set']):
                ingredients = eval(row['ingredient_set']) if isinstance(row['ingredient_set'], str) else row['ingredient_set']
                all_ingredients.update(ingredients)
        
        all_ingredients = sorted(list(all_ingredients))
        print(f"📋 Found {len(all_ingredients)} unique ingredients")
        
        # Create binary vectors for each recipe
        recipe_vectors = []
        for idx, row in recipes_df.iterrows():
            if 'ingredient_set' in row and pd.notna(row['ingredient_set']):
                ingredients = eval(row['ingredient_set']) if isinstance(row['ingredient_set'], str) else row['ingredient_set']
                vector = [1 if ingredient in ingredients else 0 for ingredient in all_ingredients]
                recipe_vectors.append(vector)
            else:
                recipe_vectors.append([0] * len(all_ingredients))
        
        recipe_vectors = np.array(recipe_vectors)
        print(f"📊 Created {len(recipe_vectors)} recipe vectors of length {len(all_ingredients)}")
        
        # Show sample data
        print(f"\n🍳 Sample Recipes:")
        for i in range(min(3, len(recipes_df))):
            recipe = recipes_df.iloc[i]
            if 'ingredient_set' in recipe:
                ingredients = eval(recipe['ingredient_set']) if isinstance(recipe['ingredient_set'], str) else recipe['ingredient_set']
                print(f"  {i+1}. {recipe['name']}: {len(ingredients)} ingredients")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def standardize_ingredient(ingredient):
    """Standardize ingredient name"""
    ingredient = str(ingredient).lower().strip()
    
    # Common mappings
    mappings = {
        'garlic': 'garlic',
        'onion': 'onion',
        'onions': 'onion',
        'tomato': 'tomato',
        'tomatoes': 'tomato',
        'ginger': 'ginger',
        'chicken': 'chicken',
        'pork': 'pork',
        'beef': 'beef',
        'shrimp': 'shrimp',
        'fish': 'fish',
        'squid': 'squid',
        'crab': 'crab',
        'coconut milk': 'coconut_milk',
        'coconutmilk': 'coconut_milk',
        'gata': 'coconut_milk',
        'fish sauce': 'fish_sauce',
        'fishsauce': 'fish_sauce',
        'chili': 'chili',
        'siling': 'chili',
        'sili': 'chili',
        'eggplant': 'eggplant',
        'okra': 'okra',
        'ampalaya': 'bitter_melon',
        'bitter melon': 'bitter_melon',
        'kangkong': 'water_spinach',
        'water spinach': 'water_spinach',
        'pechay': 'pechay',
        'papaya': 'papaya',
        'pumpkin': 'pumpkin',
        'kalabasa': 'pumpkin',
        'taro': 'taro',
        'gabi': 'taro',
        'corn': 'corn',
        'mais': 'corn',
        'banana': 'banana',
        'saba': 'banana',
        'egg': 'egg',
        'eggs': 'egg',
        'salt': 'salt',
        'pepper': 'pepper',
        'oil': 'oil',
        'water': 'water',
        'kalamansi': 'kalamansi',
        'calamansi': 'kalamansi',
        'tanglad': 'lemongrass',
        'lemongrass': 'lemongrass',
        'soy sauce': 'soy_sauce',
        'soysauce': 'soy_sauce',
        'vinegar': 'vinegar',
        'bay leaves': 'bay_leaves',
        'bayleaves': 'bay_leaves',
        'sinigang mix': 'sinigang_mix',
        'sinigangmix': 'sinigang_mix',
    }
    
    # Check for exact match first
    for key, value in mappings.items():
        if ingredient == key:
            return value
    
    # Check for partial match
    for key, value in mappings.items():
        if key in ingredient:
            return value
    
    # Check if it's already in our ingredient list
    if all_ingredients is not None:
        for std_ing in all_ingredients:
            if std_ing.lower() == ingredient:
                return std_ing
    
    # If no match found, return as is (will be filtered out)
    return ingredient

def create_user_vector(user_ingredients):
    """Create binary vector for user ingredients"""
    # Standardize user ingredients
    standardized = set()
    for ing in user_ingredients:
        std_ing = standardize_ingredient(ing)
        if std_ing in all_ingredients:
            standardized.add(std_ing)
    
    # Create binary vector
    vector = [1 if ingredient in standardized else 0 for ingredient in all_ingredients]
    return np.array(vector).reshape(1, -1), standardized

def calculate_cosine_similarity(user_vector):
    """Calculate cosine similarity between user vector and all recipes"""
    similarities = cosine_similarity(user_vector, recipe_vectors)
    return similarities.flatten()

def get_recipe_ingredients(recipe):
    """Get ingredient set for a recipe"""
    if 'ingredient_set' in recipe and pd.notna(recipe['ingredient_set']):
        ingredients = eval(recipe['ingredient_set']) if isinstance(recipe['ingredient_set'], str) else recipe['ingredient_set']
        return set(ingredients)
    return set()

def parse_ingredients_list(ingredients_str):
    """Parse ingredients string into list"""
    if pd.isna(ingredients_str) or not str(ingredients_str).strip():
        return []
    
    text = str(ingredients_str)
    items = [item.strip() for item in text.split(';') if item.strip()]
    return items

def parse_steps_list(steps_str):
    """Parse steps string into list"""
    if pd.isna(steps_str) or not str(steps_str).strip():
        return []
    
    text = str(steps_str)
    
    # Try to split by numbered steps
    import re
    steps = re.split(r'\d+\.\s*', text)
    steps = [step.strip() for step in steps if step.strip()]
    
    # If no numbered steps found, split by semicolons
    if not steps:
        steps = [step.strip() for step in text.split(';') if step.strip()]
    
    return steps

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    print("🚀 Starting Cosine Similarity Recipe Recommender...")
    if not load_and_prepare_data():
        print("❌ Failed to load data. API will not work properly.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Recipe Recommendation API (Cosine Similarity)",
        "version": "3.0.0",
        "description": "Finds similar recipes using cosine similarity on ingredient vectors",
        "endpoints": {
            "GET /": "This documentation",
            "POST /recommend": "Get recipe recommendations (above 50% similarity)",
            "GET /health": "Check API health",
            "GET /ingredients": "Get list of all ingredients",
            "GET /recipes": "Get list of all recipes",
            "GET /similarity-matrix": "Get similarity between all recipes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if recipes_df is None or all_ingredients is None or recipe_vectors is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "status": "healthy",
        "recipes_count": len(recipes_df),
        "ingredients_count": len(all_ingredients),
        "vectors_shape": recipe_vectors.shape
    }

@app.get("/ingredients")
async def get_all_ingredients():
    """Get all available ingredients"""
    if all_ingredients is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Group ingredients by category for easier browsing
    categories = {
        "Proteins": ['chicken', 'pork', 'beef', 'shrimp', 'fish', 'squid', 'crab'],
        "Vegetables": ['onion', 'garlic', 'tomato', 'ginger', 'eggplant', 'okra', 'bitter_melon', 
                      'water_spinach', 'pechay', 'papaya', 'pumpkin', 'taro', 'corn', 'banana'],
        "Sauces & Seasonings": ['fish_sauce', 'soy_sauce', 'vinegar', 'salt', 'pepper', 'chili', 
                               'kalamansi', 'lemongrass', 'bay_leaves', 'sinigang_mix'],
        "Other": ['coconut_milk', 'egg', 'oil', 'water']
    }
    
    categorized = {}
    for category, items in categories.items():
        categorized[category] = [ing for ing in all_ingredients if ing in items]
    
    # Uncategorized ingredients
    all_ing_set = set(all_ingredients)
    for items in categories.values():
        all_ing_set -= set(items)
    categorized["Other"] += list(all_ing_set)
    
    return {
        "total_ingredients": len(all_ingredients),
        "ingredients": all_ingredients,
        "categorized": categorized
    }

@app.get("/recipes")
async def get_all_recipes():
    """Get all available recipes"""
    if recipes_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    recipes_list = []
    for idx, row in recipes_df.iterrows():
        ingredients = get_recipe_ingredients(row)
        recipes_list.append({
            "name": row['name'] if 'name' in row else "Unknown",
            "cuisine": row.get('cuisine', 'Unknown'),
            "servings": row.get('servings', 'Not specified'),
            "prep_time": row.get('prep_time', 'Not specified'),
            "cook_time": row.get('cook_time', 'Not specified'),
            "total_time": row.get('total_time', 'Not specified'),
            "ingredient_count": len(ingredients)
        })
    
    return {
        "total_recipes": len(recipes_list),
        "recipes": recipes_list
    }

@app.get("/similarity-matrix")
async def get_similarity_matrix():
    """Get cosine similarity matrix between all recipes"""
    if recipe_vectors is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(recipe_vectors)
    
    # Get top 3 most similar recipes for each recipe
    similar_pairs = []
    for i in range(min(10, len(recipes_df))):  # Limit to first 10 for performance
        recipe_name = recipes_df.iloc[i]['name']
        similarities = similarity_matrix[i]
        
        # Get indices of top 3 most similar (excluding self)
        top_indices = np.argsort(similarities)[-4:-1][::-1]
        
        similar_recipes = []
        for idx in top_indices:
            if idx != i:
                similar_recipes.append({
                    "recipe": recipes_df.iloc[idx]['name'],
                    "similarity": float(similarities[idx])
                })
        
        similar_pairs.append({
            "recipe": recipe_name,
            "similar_recipes": similar_recipes
        })
    
    return {
        "total_recipes": len(recipes_df),
        "similarity_matrix_shape": similarity_matrix.shape,
        "sample_similarities": similar_pairs
    }

@app.post("/recommend", response_model=RecipeResponse)
async def recommend_recipes(request: RecipeRequest):
    """
    Get recipe recommendations based on ingredient similarity
    
    Example request:
    ```json
    {
        "ingredients": ["Chicken", "Garlic", "Onions"]
    }
    ```
    
    Uses cosine similarity to find recipes with similar ingredient profiles.
    Only returns recipes with similarity score above 50%.
    """
    if recipes_df is None or all_ingredients is None or recipe_vectors is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Validate input
    if not request.ingredients:
        raise HTTPException(status_code=400, detail="No ingredients provided")
    
    # Create user vector
    user_vector, user_ingredients_std = create_user_vector(request.ingredients)
    
    if len(user_ingredients_std) == 0:
        available = ", ".join(all_ingredients[:15])
        raise HTTPException(
            status_code=400, 
            detail=f"No valid ingredients found. Available ingredients include: {available}..."
        )
    
    # Calculate cosine similarity
    similarities = calculate_cosine_similarity(user_vector)
    
    # Filter recipes with similarity > 50% (0.5)
    threshold = 0.5
    above_threshold_indices = [i for i, score in enumerate(similarities) if score >= threshold]
    
    if not above_threshold_indices:
        return {
            "status": "success",
            "message": f"No recipes found with similarity above {threshold*100}%",
            "timestamp": datetime.now().isoformat(),
            "input_ingredients": request.ingredients,
            "recommendations": []
        }
    
    # Sort filtered recipes by similarity (highest first)
    filtered_scores = [(i, similarities[i]) for i in above_threshold_indices]
    filtered_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Prepare recommendations (all above threshold)
    recommendations = []
    
    for idx, similarity_score in filtered_scores:
        recipe = recipes_df.iloc[idx]
        
        # Get recipe ingredients
        recipe_ingredients = get_recipe_ingredients(recipe)
        
        # Find matched and missing ingredients
        matched_ingredients = list(user_ingredients_std.intersection(recipe_ingredients))
        missing_ingredients = list(recipe_ingredients - user_ingredients_std)
        
        # Parse ingredients and steps
        ingredients_list = parse_ingredients_list(recipe.get('ingredients_str', ''))
        steps_list = parse_steps_list(recipe.get('steps_str', ''))
        
        # Create recommendation
        recommendation = {
            "recipe_name": recipe['name'],
            "cuisine": str(recipe.get('cuisine', 'Unknown')),
            "servings": str(recipe.get('servings', 'Not specified')),
            "prep_time": str(recipe.get('prep_time', 'Not specified')),
            "cook_time": str(recipe.get('cook_time', 'Not specified')),
            "total_time": str(recipe.get('total_time', 'Not specified')),
            "similarity_score": round(similarity_score * 100, 2),  # Convert to percentage
            "ingredients_list": ingredients_list,
            "steps_list": steps_list,
            "matched_ingredients": matched_ingredients,
            "missing_ingredients": missing_ingredients,
            "total_ingredients": len(recipe_ingredients),
            "matched_count": len(matched_ingredients),
            "missing_count": len(missing_ingredients)
        }
        
        recommendations.append(recommendation)
    
    # Prepare response
    response = {
        "status": "success",
        "message": f"Found {len(recommendations)} recipes with similarity above {threshold*100}%",
        "timestamp": datetime.now().isoformat(),
        "input_ingredients": request.ingredients,
        "standardized_ingredients": list(user_ingredients_std),
        "recommendations": recommendations
    }
    
    return response

@app.post("/find-similar")
async def find_similar_recipes(request: RecipeRequest):
    """Find recipes similar to the input ingredients (alternative algorithm)"""
    if recipes_df is None or all_ingredients is None or recipe_vectors is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Create user vector
    user_vector, user_ingredients_std = create_user_vector(request.ingredients)
    
    # Calculate Jaccard similarity (alternative metric)
    similarities = []
    for idx, recipe in recipes_df.iterrows():
        recipe_ingredients = get_recipe_ingredients(recipe)
        
        # Calculate Jaccard similarity
        intersection = len(user_ingredients_std.intersection(recipe_ingredients))
        union = len(user_ingredients_std.union(recipe_ingredients))
        jaccard = intersection / union if union > 0 else 0
        
        # Also calculate cosine similarity
        cosine = cosine_similarity(user_vector, recipe_vectors[idx].reshape(1, -1))[0][0]
        
        # Combined score (weighted average)
        combined_score = (jaccard * 0.6) + (cosine * 0.4)
        
        similarities.append({
            "index": idx,
            "jaccard": jaccard,
            "cosine": cosine,
            "combined": combined_score
        })
    
    # Sort by combined score
    similarities.sort(key=lambda x: x["combined"], reverse=True)
    
    # Filter by threshold (50%)
    threshold = 0.5
    filtered_items = [item for item in similarities if item["combined"] >= threshold]
    
    if not filtered_items:
        return {
            "input_ingredients": request.ingredients,
            "algorithm": "Combined Jaccard + Cosine Similarity",
            "message": f"No recipes found with combined similarity above {threshold*100}%",
            "recommendations": []
        }
    
    recommendations = []
    for item in filtered_items:
        recipe = recipes_df.iloc[item["index"]]
        recipe_ingredients = get_recipe_ingredients(recipe)
        
        matched = list(user_ingredients_std.intersection(recipe_ingredients))
        missing = list(recipe_ingredients - user_ingredients_std)
        
        recommendations.append({
            "recipe_name": recipe['name'],
            "cuisine": recipe.get('cuisine', 'Unknown'),
            "combined_score": round(item["combined"] * 100, 2),
            "jaccard_score": round(item["jaccard"] * 100, 2),
            "cosine_score": round(item["cosine"] * 100, 2),
            "matched_ingredients": matched,
            "missing_ingredients": missing
        })
    
    return {
        "input_ingredients": request.ingredients,
        "algorithm": "Combined Jaccard + Cosine Similarity",
        "message": f"Found {len(recommendations)} recipes with combined similarity above {threshold*100}%",
        "recommendations": recommendations
    }

@app.post("/recommend-by-ingredient-overlap")
async def recommend_by_overlap(request: RecipeRequest):
    """Simple recommendation based on ingredient overlap count"""
    if recipes_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Standardize user ingredients
    user_ingredients_std = set()
    for ing in request.ingredients:
        std_ing = standardize_ingredient(ing)
        user_ingredients_std.add(std_ing)
    
    # Calculate overlap for each recipe
    scores = []
    for idx, recipe in recipes_df.iterrows():
        recipe_ingredients = get_recipe_ingredients(recipe)
        
        # Calculate overlap
        overlap = len(user_ingredients_std.intersection(recipe_ingredients))
        total_recipe_ingredients = len(recipe_ingredients)
        
        # Calculate percentage
        if total_recipe_ingredients > 0:
            percentage = (overlap / total_recipe_ingredients) * 100
        else:
            percentage = 0
        
        scores.append({
            "index": idx,
            "overlap_count": overlap,
            "overlap_percentage": percentage,
            "total_recipe_ingredients": total_recipe_ingredients
        })
    
    # Filter by threshold (50% overlap)
    threshold = 50
    filtered_scores = [score for score in scores if score["overlap_percentage"] >= threshold]
    
    if not filtered_scores:
        return {
            "input_ingredients": request.ingredients,
            "algorithm": "Ingredient Overlap Percentage",
            "message": f"No recipes found with ingredient overlap above {threshold}%",
            "recommendations": []
        }
    
    # Sort by overlap percentage (highest first)
    filtered_scores.sort(key=lambda x: x["overlap_percentage"], reverse=True)
    
    recommendations = []
    for item in filtered_scores:
        recipe = recipes_df.iloc[item["index"]]
        recipe_ingredients = get_recipe_ingredients(recipe)
        
        matched = list(user_ingredients_std.intersection(recipe_ingredients))
        missing = list(recipe_ingredients - user_ingredients_std)
        
        ingredients_list = parse_ingredients_list(recipe.get('ingredients_str', ''))
        steps_list = parse_steps_list(recipe.get('steps_str', ''))
        
        recommendations.append({
            "recipe_name": recipe['name'],
            "cuisine": recipe.get('cuisine', 'Unknown'),
            "servings": recipe.get('servings', 'Not specified'),
            "prep_time": recipe.get('prep_time', 'Not specified'),
            "cook_time": recipe.get('cook_time', 'Not specified'),
            "overlap_percentage": round(item["overlap_percentage"], 2),
            "overlap_count": item["overlap_count"],
            "total_recipe_ingredients": item["total_recipe_ingredients"],
            "matched_ingredients": matched,
            "missing_ingredients": missing,
            "ingredients_list": ingredients_list,
            "steps_list": steps_list
        })
    
    return {
        "input_ingredients": request.ingredients,
        "algorithm": "Ingredient Overlap Percentage",
        "message": f"Found {len(recommendations)} recipes with ingredient overlap above {threshold}%",
        "recommendations": recommendations
    }

if __name__ == "__main__":
    # Check if required file exists
    required_file = 'recipes_processed.csv'
    
    print("🔍 Checking for required file...")
    if os.path.exists(required_file):
        print(f"  ✅ {required_file}")
        
        print("\n🚀 Starting FastAPI server...")
        print("📌 Endpoints:")
        print("  - GET  /                         - API documentation")
        print("  - GET  /health                   - Health check")
        print("  - GET  /ingredients              - List all ingredients")
        print("  - GET  /recipes                  - List all recipes")
        print("  - GET  /similarity-matrix        - Get similarity between recipes")
        print("  - POST /recommend                - Cosine similarity recommendations (above 50%)")
        print("  - POST /find-similar             - Combined similarity recommendations (above 50%)")
        print("  - POST /recommend-by-ingredient-overlap - Simple overlap recommendations (above 50%)")
        print("\n📝 Example POST request to /recommend:")
        print('''{
  "ingredients": ["Chicken", "Garlic", "Onions"]
}''')
        print("\n📊 Note: Only recipes with similarity score ≥ 50% will be returned")
        
        # Start the server
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print(f"  ❌ {required_file}")
        print("\n❌ Missing file! Please run a training script first.")
        print("💡 You need recipes_processed.csv with columns:")
        print("   - name, cuisine, servings, prep_time, cook_time")
        print("   - ingredients_str, steps_str, ingredient_set")