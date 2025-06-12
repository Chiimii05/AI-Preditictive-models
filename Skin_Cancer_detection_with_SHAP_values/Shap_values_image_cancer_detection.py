# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import shap

# Loading datasets
df = pd.read_csv('input/hmnist_28_28_RGB.csv')
df_metadata = pd.read_csv('input/HAM10000_metadata.csv')


df_metadata = df_metadata.drop(columns=['dx_type', 'dx', 'image_id', 'lesion_id'])

# Extracting features and labels
features = df.loc[:, 'pixel0000':'pixel2351']
labels = df['label']

print(features.isnull().sum().sum())

# Standardizing the features (important for PCA)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# Applying PCA
pca = PCA(n_components=0.95) # You can choose 'mle', a specific number, or a float between 0 and 1
pca_features = pca.fit_transform(features_scaled)

# Creating a DataFrame for the PCA features
pca_df = pd.DataFrame(data=pca_features, columns=[f'pca{i:04}' for i in range(pca_features.shape[1])])

# Map sex to 0 1
df_metadata['sex'] = df_metadata['sex'].map({'male': 0, 'female': 1})

# One hot encode localization
onehot_encoder = OneHotEncoder(sparse=False)
localization_encoded = onehot_encoder.fit_transform(df_metadata[['localization']])
localization_encoded_df = pd.DataFrame(localization_encoded, columns=onehot_encoder.get_feature_names_out(['localization']))

# Concatenate the new DataFrame to the original one (minus the original "localization" column)
df_metadata = pd.concat([df_metadata.drop('localization', axis=1), localization_encoded_df], axis=1)

# Concatenating the PCA features with the labels
pca_df = pd.concat([pca_df, labels.reset_index(drop=True)], axis=1)

# Concatenating the PCA features with the labels
final_df = pd.concat([df_metadata, pca_df], axis=1)

final_df

# Analyzing label inbalance
print("Class Distribution:\n", final_df['label'].value_counts())

# Check for missing values or errors
print(final_df.isna().sum(axis=0))

# Not many nans so lets remove
final_df = final_df.dropna()

# Check again
print(final_df.isna().sum(axis=0))

"""# Balancing Techniques SMOTE vs SMOTEENN vs LABEL WEIGHT
Testing only on pca_df for easy of smote without categoricals
"""

# Separating features and labels
X = pca_df.drop('label', axis=1)
y = pca_df['label']

# Splitting the original dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying SMOTE and SMOTEENN only to the training data
smote = SMOTE(random_state=42)
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)

smoteenn = SMOTEENN(random_state=42)
X_train_SMOTEENN, y_train_SMOTEENN = smoteenn.fit_resample(X_train, y_train)

print(X_train)

print("Class Distribution after SMOTE:\n", y_train_SMOTE.value_counts()) # inbalance is fixed
print("Class Distribution after SMOTEENN:\n", y_train_SMOTEENN.value_counts()) # inbalance is fixed
# The NN steps detect values that have an overlap with other classes. We think that the last label since the number is reduced significantly
# means that there is a lot of overlaps

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = rf.predict(X_test)
report_noSMOTE = classification_report(y_test, predictions, output_dict=True)
overall_f1_noSMOTE = report_noSMOTE['weighted avg']['f1-score']
accuracy_noSMOTE = report_noSMOTE['accuracy']
print(f"Overall without SMOTE F1 Score, Accuracy: {overall_f1_noSMOTE}, {accuracy_noSMOTE}")

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_SMOTE, y_train_SMOTE)

# Make predictions and evaluate the model
predictions = rf.predict(X_test)
report_SMOTE = classification_report(y_test, predictions, output_dict=True)
overall_f1_SMOTE = report_SMOTE['weighted avg']['f1-score']
accuracy_SMOTE = report_SMOTE['accuracy']
print(f"Overall using SMOTE F1 Score, Accuracy: {overall_f1_SMOTE}, {accuracy_SMOTE}")

# Initialize and train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_SMOTEENN, y_train_SMOTEENN)

# Make predictions and evaluate the model
predictions = rf.predict(X_test)
report_SMOTEEN = classification_report(y_test, predictions, output_dict=True)
overall_f1_SMOTEENN = report_SMOTEEN['weighted avg']['f1-score']
accuracy_SMOTEENN = report_SMOTEEN['accuracy']
print(f"Overall using SMOTEENN F1 Score, Accuracy: {overall_f1_SMOTEENN}, {accuracy_SMOTEENN}")

# Initialize the Random Forest model with class weights
rf_class_weight = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_class_weight.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions_class_weight = rf_class_weight.predict(X_test)
report_RANDOMFORE = classification_report(y_test, predictions, output_dict=True)
overall_f1_RANDOMFORE = report_RANDOMFORE['weighted avg']['f1-score']
accuracy_RANDOMFORE = report_RANDOMFORE['accuracy']
print(f"Overall using Balanced Weight F1 Score, Accuracy: {overall_f1_RANDOMFORE}, {accuracy_RANDOMFORE}")

accuracy_baseline = overall_f1_noSMOTE
accuracy_smote = overall_f1_SMOTE
accuracy_smoteenn = overall_f1_SMOTEENN
accuracy_classweight = overall_f1_RANDOMFORE
# Model names
models = ['Baseline', 'SMOTE', 'SMOTEENN', 'CLASS WEIGHT']

# Accuracy values
accuracies = [accuracy_baseline, accuracy_smote, accuracy_smoteenn, accuracy_classweight]

# Creating the bar chart
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['grey', 'blue', 'lightblue', 'orange'])

plt.title('Model Accuracy Comparison')
plt.ylabel('F1 Score')
plt.ylim(0.5, 0.7)  # Adjust based on your actual accuracy values for better visualization
plt.xlabel('Model')
plt.xticks(models)

# Adding the accuracy values on top of each bar
for i, accuracy in enumerate(accuracies):
    plt.text(i, accuracy + 0.005, f'{accuracy:.2f}', ha='center')

# Show plot
plt.tight_layout()
plt.show()

"""# MODELS

using final_df with all characteristics and no oversampling
"""

# Separating features and labels
X = final_df.drop('label', axis=1)
y = final_df['label']

# Defining the models
models = {
    "MLPClassifier": MLPClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Bagging" : BaggingClassifier(base_estimator=RandomForestClassifier(), random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Training
    predictions = model.predict(X_test)  # Predicting
    print(f"Evaluation of {name}:\n{classification_report(y_test, predictions)}")

from sklearn.ensemble import VotingClassifier

# Assuming logistic_regression, random_forest, and gradient_boosting are your best models
voting_clf = VotingClassifier(
    estimators=[
        ('knn', models["K-Nearest Neighbors"]),
        ('rf', models["Random Forest"]),
        ('mlp', models["MLPClassifier"]),
        ('gb', models["Gradient Boosting"])
    ],
    voting='soft'  # Use 'soft' if you prefer averaging probabilities instead of majority voting
)

voting_clf.fit(X_train_SMOTE, y_train_SMOTE)

predictions = voting_clf.predict(X_test)
print("Evaluation of Voting Classifier with Pre-trained Models:\n", classification_report(y_test, predictions))



# Assuming logistic_regression, random_forest, and gradient_boosting are your best models
voting_clf = VotingClassifier(
    estimators=[
        ('knn', models["K-Nearest Neighbors"]),
        ('rf', models["Random Forest"]),
        ('mlp', models["MLPClassifier"]),
        ('gb', models["Gradient Boosting"])
    ],
    voting='hard'  # Use 'hard' if you prefer majority voting instead of averaging probabilities
)

voting_clf.fit(X_train_SMOTE, y_train_SMOTE)

predictions = voting_clf.predict(X_test)
print("Evaluation of Voting Classifier with Pre-trained Models:\n", classification_report(y_test, predictions))

def plot_image(data_row):
    # Assuming pixel data columns are named from 'pixel0000' to 'pixel2351'
    # Extract only the columns that contain pixel data
    pixel_columns = [col for col in data_row.index if col.startswith('pixel')]
    image_data = np.array(data_row[pixel_columns], dtype=np.uint8).reshape(28, 28, 3)

    # Plot the image
    plt.imshow(image_data)
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

plot_image(features.iloc[9])

# Assuming 'scaler' and 'pca' are already trained as per your previous script
# Also assuming 'onehot_encoder' and the mapping for 'sex' are defined

def process_instance(features_image, scaler=scaler, pca=pca):
    # Reshape from (28, 28, 3) to (3, 28, 28) and flatten to 1D array

    features_flattened = features_image.reshape(features_image.shape[0], 28*28*3)
    features_df = pd.DataFrame(features_flattened, columns=[f'pixel{i:04d}' for i in range(28*28*3)])
    # Standardizing the features for the instance and applying PCA
    features_scaled_instance = scaler.transform(features_df)
    pca_features_instance = pca.transform(features_scaled_instance)

    # Prepare the PCA features as a DataFrame
    pca_df = pd.DataFrame(data=pca_features_instance, columns=[f'pca{i:04}' for i in range(pca_features_instance.shape[1])])

    return pca_df

def reshape_to_image(flattened_features):
    # Assuming the input is a flat array of length 2352 (3 * 28 * 28)
    return flattened_features.values.reshape(-1, 28, 28, 3)

def predict(x, model=models["Gradient Boosting"]):
    # Get the prediction from the model
    prediction = model.predict(process_instance(x))

    return prediction


print(predict(reshape_to_image(features.iloc[9])))

random_indices = np.random.choice(df.index, size=16, replace=False)
image = reshape_to_image(features.loc[random_indices])
print(image.shape)

masker_names=["blur(28,28)","blur(14,14)","inpaint_telea","inpaint_ns"]

for masker_name in masker_names:
    masker = shap.maskers.Image(masker_name, shape = (28, 28, 3))
    explainer = shap.Explainer(predict, masker)
    shap_values = explainer(image, max_evals=8192, batch_size=64, silent=True)
    print(f"Masker: {masker_name}")
    shap.image_plot(shap_values, pixel_values=image/255)

print(predict(image))
print(df.loc[random_indices]["label"].values)