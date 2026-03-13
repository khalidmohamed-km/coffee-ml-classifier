import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.array([
    [9, 1, 0],
    [8, 2, 0],
    [3, 6, 1],
    [2, 7, 1],
    [4, 4, 3],
    [5, 3, 4]
])

y = np.array([0, 0, 1, 1, 2, 2])

model = DecisionTreeClassifier()
model.fit(X, y)

new_coffee = [[0, 7, 9]]
prediction = model.predict(new_coffee)
coffee_names = {0: "Espresso", 1: "Latte", 2: "Cappuccino"}

print("Prediction:", coffee_names[prediction[0]])
