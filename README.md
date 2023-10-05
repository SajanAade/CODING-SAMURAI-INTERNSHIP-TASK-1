# CODING-SAMURAI-INTERNSHIP-TASK-1

# Iris-Flower-Classification-Dataset
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but KNeighborsClassifier was fitted with feature names")

# Step 1: Pre-process the dataset
from sklearn.model_selection import train_test_split
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Choose a machine learning algorithm; For this example, let's use the k-nearest neighbors (KNN) algorithm.
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Step 3: Train the model
knn_classifier.fit(X_train, y_train)

# Step 4: Evaluate the model
from sklearn.metrics import accuracy_score
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 5: Make predictions on new data
# Provide sepal length, sepal width, petal length, petal width
new_data = [[5.1, 3.5, 1.4, 0.2]]  
predicted_species = knn_classifier.predict(new_data)
print(f"Predicted species: {predicted_species[0]}")
