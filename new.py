import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import optuna

# Load data
df = pd.read_csv("train_data1 (1).csv")
df = df.dropna(subset=["What would you like to become when you grow up"])
df = df.drop(['Leadership Experience', 'Timestamp'], axis=1)

X = df.drop('What would you like to become when you grow up', axis=1)
y = df['What would you like to become when you grow up']

# Define features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Info Gain Feature Selection
X_processed = preprocessor.fit_transform(X)
info_gain = mutual_info_classif(X_processed, y, discrete_features='auto')
cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)
all_feature_names = list(numeric_features) + list(cat_feature_names)
info_df = pd.DataFrame({'Feature': all_feature_names, 'Information_Gain': info_gain}).sort_values(by='Information_Gain', ascending=False)

# Filter top features
threshold = 0.001
selected_features = info_df[info_df['Information_Gain'] > threshold]['Feature'].tolist()
X_transformed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
                                columns=all_feature_names)
X_selected = X_transformed_df[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    model_name = trial.suggest_categorical("model", ["RandomForest", "GradientBoosting"])
    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 150),
            max_depth=trial.suggest_int("max_depth", 5, 30),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            random_state=42
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 150),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10)
        )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

# Run optimization
@st.cache_resource
def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    return study.best_params

best_params = run_optuna()

# Train final model
if best_params["model"] == "RandomForest":
    model = RandomForestClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        random_state=42
    )
else:
    model = GradientBoostingClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        min_samples_split=best_params["min_samples_split"]
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Streamlit UI
st.title("Career Prediction App 🎓")
st.subheader("Predict what someone wants to become when they grow up")

# User Input
user_data = {}
for col in numeric_features:
    user_data[col] = st.number_input(f"{col}", value=float(X[col].mean()))

for col in categorical_features:
    options = df[col].dropna().unique().tolist()
    user_data[col] = st.selectbox(f"{col}", options)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_data])
    processed_input = preprocessor.transform(input_df)
    processed_input_df = pd.DataFrame(processed_input.toarray() if hasattr(processed_input, "toarray") else processed_input,
                                      columns=all_feature_names)
    selected_input = processed_input_df[selected_features]
    prediction = model.predict(selected_input)[0]
    st.success(f"🎯 Predicted Career: **{prediction}**")

st.markdown("---")
st.header("Model Performance")
st.metric("Accuracy", f"{acc:.2f}")
st.metric("Precision", f"{precision:.2f}")
st.metric("Recall", f"{recall:.2f}")
st.metric("F1 Score", f"{f1:.2f}")
