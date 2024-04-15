import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
df = pd.read_csv(r"C:\Users\sushu\OneDrive\Desktop\OTHER COURSES\Intern\CodeAlpha\Machine Learning\Iris Flower\datacsv.csv")
df.dropna(inplace = True)
species = df['species'].value_counts().reset_index()
plt.figure(figsize=(8,8))
plt.pie(species['count'],labels=['Iris-setosa','Iris-versicolor','Iris-virginica'],autopct='%1.3f%%')
plt.legend(loc='upper left')
plt.show()
sns.scatterplot(x='sepal_length', y='sepal_width', data=df, hue='species')
plt.title('Sepal Width vs Sepal Length')
plt.show()
sns.scatterplot(x='petal_length', y='petal_width', data=df, hue='species')
plt.title('petal Width vs petal Length')
plt.show()
# Split the dataset into features (X) and target labels (y)
X = df.drop(columns=["species"])  # Features
y = df["species"]  # Target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=df["species"].unique()))