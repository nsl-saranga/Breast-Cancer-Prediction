import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Read the data
df = pd.read_csv('data.csv')

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 32'])

# Checking for nulls
print(df.isnull().sum())

# Checking for duplicates
print("No. duplicated rows: ", df.duplicated().sum())

# Separating features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Data visualization
benign_df = X[y == 'B']
malignant_df = X[y == 'M']

# Plotting
plt.figure(figsize=(8, 6))

# Scatter plot for benign cases
plt.scatter(benign_df.index, benign_df['area_se'], color='skyblue', label='Benign')

# Scatter plot for malignant cases
plt.scatter(malignant_df.index, malignant_df['area_se'], color='salmon', label='Malignant')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Compactness SE')
plt.title('Compactness SE Distribution by Diagnosis')
plt.legend()

plt.grid(True)
plt.show()

# feature selection
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
rfe.fit(X,y)

# Get the selected features
selected_features_rfe = X.columns[rfe.support_]
print("Selected features using RFE with Decision Tree:", selected_features_rfe, rfe.ranking_)

for i, col in zip(range(X.shape[1]), X.columns):
    print(f"{col} selected={rfe.support_[i]} rank={rfe.ranking_[i]}")

print("selected features:", selected_features_rfe)

X = X[selected_features_rfe]
print(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# converting labels to binary
y = y.apply(lambda x: 1 if x == 'M' else 0)

model = svm.SVC()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model.fit(X_train, y_train)

# get prediction results
y_pred = model.predict(X_test)

# evaluate the model.
acc = accuracy_score(y_test, y_pred)
class_rep = classification_report(y_test, y_pred)

print("Accuracy score : ", str(acc))
print("Classification Report : \n", str(class_rep))

joblib.dump(model, "cancer_prediction_model.joblib")
joblib.dump(scaler, "scaler.joblib")




