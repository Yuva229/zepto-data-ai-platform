import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv('titanic.csv')

# 7. STRATIFIED SPLIT (BEFORE PREPROCESSING)
X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 8. PREPROCESSING PIPELINE
numeric_features = ['age', 'fare']
categorical_features = ['sex', 'embarked']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 9. TRAIN MODELS
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'accuracy': clf.score(X_test, y_test),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'report': classification_report(y_test, y_pred)
    }
    print(f"\n{name} Results:")
    print(results[name]['report'])

# 12. HYPERPARAMETER TUNING
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__max_features': ['sqrt', 'log2']
}
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                              ('classifier', RandomForestClassifier(oob_score=True, random_state=42))])
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"\nBest Params: {grid_search.best_params_}")
print(f"OOB Score: {grid_search.best_estimator_.named_steps['classifier'].oob_score_}")

# 15. SAVE PIPELINE
joblib.dump(grid_search.best_estimator_, 'best_titanic_pipeline.pkl')
print("\nSaved best pipeline as best_titanic_pipeline.pkl")
