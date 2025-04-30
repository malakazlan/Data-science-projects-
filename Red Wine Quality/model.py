import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv('winequality-red.csv')
print('First 5 rows:')
print(df.head())

# 2. Data Overview & EDA
print('\nData Info:')
df.info()
print('\nStatistical Summary:')
print(df.describe())
print('\nMissing Values:')
print(df.isnull().sum())

# Distribution of target variable
plt.figure()
sns.countplot(x='quality', data=df)
plt.title('Wine Quality Distribution')
plt.savefig('quality_distribution.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Data Processing
# Convert quality to binary classification: Good (>=7), Not Good (<7)
df['good'] = (df['quality'] >= 7).astype(int)
plt.figure()
sns.countplot(x='good', data=df)
plt.title('Good vs Not Good Wine')
plt.savefig('good_vs_notgood.png')
plt.close()

# Features and target
X = df.drop(['quality', 'good'], axis=1)
y = df['good']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training & Selection
models = {
    'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_report': classification_report(y_train, y_pred_train, output_dict=True),
        'test_report': classification_report(y_test, y_pred_test, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred_test)
    }
    print(f'\n{name} - Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title(f'{name} Confusion Matrix (Test)')
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.close()
    print(classification_report(y_test, y_pred_test))
# Find best model
best_model = max(results, key=lambda k: results[k]['test_acc'])
print(f'\nBest Model: {best_model}')

# 5. Hyperparameter Tuning for Best Model
if best_model == 'RandomForest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    print('Best Params:', grid.best_params_)
    best_rf = grid.best_estimator_
    y_pred_train = best_rf.predict(X_train_scaled)
    y_pred_test = best_rf.predict(X_test_scaled)
    print('Train Accuracy:', accuracy_score(y_train, y_pred_train))
    print('Test Accuracy:', accuracy_score(y_test, y_pred_test))
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title('RandomForest Confusion Matrix (Test)')
    plt.savefig('RandomForest_best_confusion_matrix.png')
    plt.close()
    print(classification_report(y_test, y_pred_test))
# Add similar tuning for other models if they are best 
if best_model == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2, 0.5],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 1.5, 2, 3]
    }
    grid = GridSearchCV(
        XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)
    print('Best Params:', grid.best_params_)
    best_xgb = grid.best_estimator_
    y_pred_train = best_xgb.predict(X_train_scaled)
    y_pred_test = best_xgb.predict(X_test_scaled)
    print('Train Accuracy:', accuracy_score(y_train, y_pred_train))
    print('Test Accuracy:', accuracy_score(y_test, y_pred_test))
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title('XGBoost Best Confusion Matrix (Test)')
    plt.savefig('XGBoost_best_confusion_matrix.png')
    plt.close()
    print(classification_report(y_test, y_pred_test))