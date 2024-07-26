import warnings
warnings.filterwarnings('ignore')


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data (you should have your own dataset)
data = {
    'Budget': [1000000, 2000000, 3000000, 4000000, 5000000],
    'Marketing': [50000, 100000, 200000, 300000, 400000],
    'Director_Rating': [8.2, 7.9, 8.5, 7.8, 8.0],
    'Actor_Rating': [7.9, 8.0, 7.5, 8.2, 8.1],
    'Movie_Rating': [8.0, 7.8, 8.2, 7.6, 7.9]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[['Budget', 'Marketing', 'Director_Rating', 'Actor_Rating']]
y = df['Movie_Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model using the training data
model.fit(X_train, y_train)

# Predict movie ratings for the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Example prediction for a new movie
new_movie = [[2500000, 150000, 8.0, 7.7]]
predicted_rating = model.predict(new_movie)
print(f'Predicted Rating for the new movie: {predicted_rating[0]}')


