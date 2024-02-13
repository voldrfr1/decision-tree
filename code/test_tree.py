from sklearn.datasets import make_classification
import pandas as pd
from sklearn.model_selection import train_test_split
from tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from random_forrest import RandomForrest
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,        # Number of samples
    n_features=3,         # Number of features
    n_informative=2,      # Number of informative features
    n_redundant=1,          # Number of redundant features
    n_clusters_per_class=2, # Number of clusters per class
    random_state=42         # Random seed for reproducibility
)

# Create a DataFrame for the dataset
columns = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=columns)
df['target'] = y

# Display the dataset
print(df.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Decision Tree
DT = DecisionTree(50)
DT.fit(X_train, y_train)
accuracy_dt = DT.evaluate(X_test, y_test)
print(f"Decision Tree Accuracy: {accuracy_dt}")

# Scikit-learn Decision Tree
scikit_tree = DecisionTreeClassifier().fit(X_train, y_train)
y_pred_scikit = scikit_tree.predict(X_test)
accuracy_scikit = accuracy_score(y_test, y_pred_scikit)
print(f"Scikit-learn Decision Tree Accuracy: {accuracy_scikit}")

# Random Forest
RF = RandomForrest(100)
RF.fit(X_train, y_train)
y_pred_rf = RF.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf}")

# Scikit-learn Random Forest
RF_scikit = RandomForestClassifier(criterion='entropy')
RF_scikit.fit(X_train, y_train)
y_pred_rf_scikit = RF_scikit.predict(X_test)
accuracy_rf_scikit = accuracy_score(y_test, y_pred_rf_scikit)
print(f"Scikit-learn Random Forest Accuracy: {accuracy_rf_scikit}")
