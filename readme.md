# Recipe Recommendation System

This project develops an intelligent recipe recommendation system that helps users discover cooking recipes based on available ingredients while considering practical constraints like cooking time and recipe complexity. The system processes the Recipe1M+ dataset to create personalized cooking suggestions that match users' capabilities and resources.

## Project Overview

The development process followed two main phases:
1. Data preprocessing
2. Recommendation system implementation

### Data Preprocessing

The preprocessing phase focused on transforming data into meaningful features for accurate recommendations. Two main functions were developed:

#### Recipe Complexity Analysis
The `count_recipe_steps()` function analyzes recipe directions and determines their complexity:
- Uses ast library for safe string-to-object conversion
- Implements step counting for complexity measurement
- Includes robust error handling
- Creates clear metrics for recipe difficulty

#### Cooking Time Extraction
The `extract_cooking_times()` function standardizes cooking durations through:
- Regular expressions for various time formats
- Conversion system for standardizing to minutes
- Logic for summing multiple time mentions
- Specific handling for edge cases

### The Recipe Recommendation System

The system employs a two-step process:
1. Converting ingredients into numerical representations
2. Finding similarities between these representations

#### Key Features
- Uses TF-IDF vectorization for ingredient representation
- Implements cosine similarity for recipe matching
- Weights ingredients based on uniqueness (e.g., common salt vs. rare saffron)
- Handles recipes with different numbers of ingredients

### Implementation Challenges and Solutions

1. **Data Quality Issues**
   - Handled inconsistencies in cooking time representations
   - Filtered recipes with cooking times over 300 minutes (4.3% of dataset)

2. **Text Processing Complexity**
   - Developed comprehensive regex patterns for time formats
   - Created standardized time_conversions dictionary

3. **Performance Considerations**
   - Optimized TF-IDF vectorization
   - Implemented matrix storage for efficient recommendations

### Web Interface

The project includes a React.js web interface that:
- Makes the system accessible to non-technical users
- Provides visual, interactive recipe exploration
- Demonstrates system capabilities effectively

## Future Enhancements

### Advanced Embedding Options
- Implement Word2Vec for semantic relationships
- Integrate OpenAI embeddings
- Train custom embeddings for ingredient substitutions

### Enhanced Feature Integration
- Add cooking techniques analysis
- Include seasonal ingredient availability
- Incorporate nutritional profiles
- Consider cultural cuisine patterns
- Add user dietary preferences

### Improved Similarity Calculations
- Weight ingredients by role (main vs. seasoning)
- Consider quantities and proportions
- Account for ingredient substitutions
- Factor in user cooking skill level

## Installation and Setup

Clone the repository:
```bash
git clone git@github.com:tenlhak/anyfeast.git
```

## Project Structure
- `preprocess.py`: Data preprocessing functions
- `preprocess.ipynb`: Preprocessing notebook
- `vocab.py`: Vocabulary processing
- `recipe_vocabulary.csv`: Processed vocabulary data
- `requirements.txt`: Project dependencies

## License
See LICENSE file for details.

## Contributing
Please read the contributing guidelines in CONTRIBUTING.md.
