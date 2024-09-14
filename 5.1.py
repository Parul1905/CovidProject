# Importing necessary libraries
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Sample data: Features (Temperature, Humidity) and Labels (Go Outside or Stay Inside)
# Temperature is in degrees Celsius, Humidity is in percentage
X = np.array([[25, 70], [30, 60], [15, 85], [20, 50], [28, 80], [10, 90]])
y = np.array(['Yes', 'Yes', 'No', 'Yes', 'No', 'No'])

# Creating a Decision Tree Classifier model
model = DecisionTreeClassifier()

# Training the model
model.fit(X, y)

# Making a prediction for a new situation
new_situation = np.array([[22, 75]])  # Temperature: 22°C, Humidity: 75%
prediction = model.predict(new_situation)
print(f"Prediction for new situation: {prediction[0]}")

# Visualizing the Decision Tree
plt.figure(figsize=(10, 6))
tree.plot_tree(model, feature_names=['Temperature', 'Humidity'], class_names=['Yes', 'No'], filled=True)
plt.title('Decision Tree for Going Outside')
plt.show()
