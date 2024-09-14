# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: X (input features) and y (target values)
X = np.array([[0], [1], [2], [3], [4]])  # Input features (e.g., number of rooms)
y = np.array([0, 7, 14, 21, 28])           # Target values (e.g., house prices)

# Creating a Linear Regression model
model = LinearRegression()

# Fitting the model to the data
model.fit(X, y)

# Making predictions
X_new = np.array([[7]])  # New input features to predict
y_pred = model.predict(X_new)       # Predicting target values for new data
print("The prediction: ",y_pred)

# Plotting the results
plt.scatter(X, y, color='blue', label='Data points')  # Original data
plt.plot(X, model.predict(X), color='red', label='Regression line')  # Best-fit line
plt.scatter(X_new, y_pred, color='green', label='Predictions')  # Predictions
# plt.xlabel('Input feature (e.g., number of rooms)')
# plt.ylabel('Target value (e.g., house price)')
plt.legend()
plt.title('Linear Regression Example')
plt.show()
