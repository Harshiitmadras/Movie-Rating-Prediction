# Movie-Rating-Prediction

## Project Overview
This project aims to build a predictive model to estimate movie ratings based on various attributes. The dataset consists of Indian movies and includes features such as genre, director, cast, and other movie-related information. The goal is to preprocess the data, engineer relevant features, and train a machine learning model to predict movie ratings accurately.

## Dataset
The project uses two primary datasets:
- `02_IMDb Movies India.csv`: Contains information about Indian movies from IMDb.
- `Indian_Movie_Rating_Prediction.csv`: Processed dataset used for training the model.

## Methodology
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Normalizing numerical features
   
2. **Feature Engineering**
   - Extracting director success rate
   - Computing the average rating of similar movies
   - Generating additional relevant features

3. **Model Training & Evaluation**
   - Training different machine learning models (e.g., Linear Regression, Random Forest, XGBoost)
   - Evaluating performance using metrics such as RMSE and R-squared
   - Selecting the best model based on accuracy

## Requirements
To run this project, install the following dependencies:
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage
1. Open `02_Movie Rating Prediction with Python.ipynb` in Jupyter Notebook.
2. Run the cells to preprocess the data and train the model.
3. The trained model is saved as `Movie_Rating_Prediction.sav`.
4. Use the saved model to make predictions on new movie data.

## Results
The final model is evaluated based on RMSE and R-squared scores, demonstrating its effectiveness in predicting movie ratings.

## Repository Structure
```
📂 02_Indian_Movies-Rating-Prediction-master
├── 📄 02_IMDb Movies India.csv
├── 📄 Indian_Movie_Rating_Prediction.csv
├── 📄 02_Movie Rating Prediction with Python.ipynb
├── 📄 Movie_Rating_Prediction.sav
├── 📄 README.md
```

## License
This project is for educational purposes only.

## Contributor
- **Harsh Yadav**

