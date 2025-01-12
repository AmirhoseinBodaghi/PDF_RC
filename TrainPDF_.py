import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib


# Step 1: Function to parse a single file and extract data
def parse_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        content = file.read()
        regions = re.split(r'Region ID: \d+', content)
        for region in regions[1:]:
            classification_match = re.search(r'Classification: (\w+)', region)
            features_match = re.search(r"Features: ({.*})", region)
            if classification_match and features_match:
                classification = classification_match.group(1)
                features = json.loads(features_match.group(1).replace("'", '"'))
                data.append({"classification": classification, **features})
    return data

"""
def parse_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        content = file.read()
        regions = re.split(r'Region ID: \d+', content)
        print(f"Regions found in {file_path}: {len(regions) - 1}")  # Debug: Check number of regions
        for i, region in enumerate(regions[1:], start=1):
            classification_match = re.search(r'Classification:\s*(\w+)', region)
            features_match = re.search(r"Features:\s*({.*})", region)
            if classification_match and features_match:
                classification = classification_match.group(1)
                try:
                    features = json.loads(features_match.group(1).replace("'", '"'))
                    data.append({"classification": classification, **features})
                except json.JSONDecodeError as e:
                    print(f"JSON decode error in region {i}: {e}\nRegion content:\n{region}")
            else:
                print(f"Failed to parse region {i}:\n{region}")
    return data
"""


# Step 2: Load data from all text files
def load_data_from_directory(directory):
    all_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            all_data.extend(parse_file(file_path))
    return pd.DataFrame(all_data)

# Step 3: Load data
data_directory = 'C:\\Users\\e16011413\\OneDrive - Ulster University\\Desktop\\ARC\\Pdf_Project\\TrainSamples' #for running at uni laptop
# data_directory = "E:\\Py64\\env_1\\Scripts\\TrainSamples" #for running at home laptop
data = load_data_from_directory(data_directory)

# Map encoded labels back to their original string values
data['classification'] = data['classification'].astype('category')
label_mapping = dict(enumerate(data['classification'].cat.categories))
print("Label Mapping:", label_mapping)

# Encode target labels as numbers
data['classification'] = data['classification'].astype('category').cat.codes

# Step 4: Print class distribution
print("Class distribution before removing insufficient samples:")
print(data['classification'].value_counts())

# Remove classes with fewer than n_neighbors+1 samples
n_neighbors = 5
class_counts = data['classification'].value_counts()
classes_to_keep = class_counts[class_counts > n_neighbors].index
data = data[data['classification'].isin(classes_to_keep)]

# Print class distribution after removal
print("Class distribution after removing insufficient samples:")
print(data['classification'].value_counts())

# Separate features and target
X = data.drop(columns=['classification'])
y = data['classification']

# Step 5: Feature selection
selector = SelectKBest(score_func=f_classif, k='all')
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
X = X[selected_features]

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance training data
smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Print class distribution after SMOTE
print("Class distribution after SMOTE:\n", pd.Series(y_train).value_counts())

# Step 7: Train a classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Step 9: Save the model and selected features
joblib.dump(selected_features, 'selected_features.pkl')
joblib.dump(classifier, 'classification_model.pkl')


# Load the saved selected features
selected_features = joblib.load('selected_features.pkl')
print("Selected Features:")
print(selected_features)

# Load the saved classifier
classification_model = joblib.load('classification_model.pkl')
print("\nClassification Model:")
print(classification_model)

print("Model training complete. Saved classifier and selected features.")
