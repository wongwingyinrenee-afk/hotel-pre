import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('resort_df.csv')
df = df.drop('hotel_Resort Hotel', axis=1)


X = df.drop('is_canceled', axis=1)
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
print(f"Class distribution - Test: {y_test.value_counts().to_dict()}")


minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)

standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)


results = {}
feature_importances = {}


print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_minmax, y_train)

y_pred_lr = lr.predict(X_test_minmax)
y_prob_lr = lr.predict_proba(X_test_minmax)[:, 1]

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_lr),
    'f1': f1_score(y_test, y_pred_lr),
    'auc': roc_auc_score(y_test, y_prob_lr)
}


lr_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': np.abs(lr.coef_[0])
}).sort_values('Importance', ascending=False)
feature_importances['Logistic Regression'] = lr_importance

print(f"Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
print(f"F1-Score: {results['Logistic Regression']['f1']:.4f}")
print(f"ROC AUC: {results['Logistic Regression']['auc']:.4f}")
print("\nTop 10 Most Important Features:")
print(lr_importance.head(10).to_string(index=False))


print("\n" + "="*50)
print("DECISION TREE")
print("="*50)

dt = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_leaf=100,
    class_weight='balanced',
    random_state=42
)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, y_pred_dt),
    'f1': f1_score(y_test, y_pred_dt),
    'auc': roc_auc_score(y_test, y_prob_dt)
}


dt_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': dt.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importances['Decision Tree'] = dt_importance

print(f"Accuracy: {results['Decision Tree']['accuracy']:.4f}")
print(f"F1-Score: {results['Decision Tree']['f1']:.4f}")
print(f"ROC AUC: {results['Decision Tree']['auc']:.4f}")
print("\nTop 10 Most Important Features:")
print(dt_importance.head(10).to_string(index=False))


print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'f1': f1_score(y_test, y_pred_rf),
    'auc': roc_auc_score(y_test, y_prob_rf)
}


rf_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importances['Random Forest'] = rf_importance

print(f"Accuracy: {results['Random Forest']['accuracy']:.4f}")
print(f"F1-Score: {results['Random Forest']['f1']:.4f}")
print(f"ROC AUC: {results['Random Forest']['auc']:.4f}")
print("\nTop 10 Most Important Features:")
print(rf_importance.head(10).to_string(index=False))


print("\n" + "="*50)
print("XGBOOST")
print("="*50)

xgb = XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=300,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

results['XGBoost'] = {
    'accuracy': accuracy_score(y_test, y_pred_xgb),
    'f1': f1_score(y_test, y_pred_xgb),
    'auc': roc_auc_score(y_test, y_prob_xgb)
}


xgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb.feature_importances_
}).sort_values('Importance', ascending=False)
feature_importances['XGBoost'] = xgb_importance

print(f"Accuracy: {results['XGBoost']['accuracy']:.4f}")
print(f"F1-Score: {results['XGBoost']['f1']:.4f}")
print(f"ROC AUC: {results['XGBoost']['auc']:.4f}")
print("\nTop 10 Most Important Features:")
print(xgb_importance.head(10).to_string(index=False))


print("\n" + "="*50)
print("NEURAL NETWORK (MLP)")
print("="*50)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)
mlp.fit(X_train_standard, y_train)

y_pred_mlp = mlp.predict(X_test_standard)
y_prob_mlp = mlp.predict_proba(X_test_standard)[:, 1]

results['MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred_mlp),
    'f1': f1_score(y_test, y_pred_mlp),
    'auc': roc_auc_score(y_test, y_prob_mlp)
}


print("Calculating permutation importance for MLP ")
perm_importance = permutation_importance(
    mlp, X_test_standard, y_test,
    n_repeats=10,
    random_state=42,
    scoring='roc_auc'
)
mlp_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)
feature_importances['MLP'] = mlp_importance

print(f"Accuracy: {results['MLP']['accuracy']:.4f}")
print(f"F1-Score: {results['MLP']['f1']:.4f}")
print(f"ROC AUC: {results['MLP']['auc']:.4f}")
print("\nTop 10 Most Important Features:")
print(mlp_importance.head(10).to_string(index=False))


print("\n" + "="*50)
print("MODEL COMPARISON (Same Test Set)")
print("="*50)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('auc', ascending=False)
print(results_df)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, metric in enumerate(['accuracy', 'f1', 'auc']):
    results_df[metric].plot(kind='barh', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{metric.upper()} Comparison')
    axes[idx].set_xlabel(metric.capitalize())
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
models_probs = {
    'Logistic Regression': y_prob_lr,
    'Decision Tree': y_prob_dt,
    'Random Forest': y_prob_rf,
    'XGBoost': y_prob_xgb,
    'MLP': y_prob_mlp
}

for name, y_prob in models_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - All Models')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()




fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum', 'lightsalmon']

for idx, model_name in enumerate(model_names):
    importance_df = feature_importances[model_name].head(15)
    axes[idx].barh(importance_df['Feature'], importance_df['Importance'], color=colors[idx])
    axes[idx].set_xlabel('Importance')
    axes[idx].set_title(f'Top 15 Features - {model_name}')
    axes[idx].invert_yaxis()


fig.delaxes(axes[5])
plt.tight_layout()
plt.show()




all_features = set()
for model_importance in feature_importances.values():
    all_features.update(model_importance.head(20)['Feature'].values)


comparison_data = []
for feature in all_features:
    feature_row = {'Feature': feature}
    for model_name, importance_df in feature_importances.items():
        feature_importance = importance_df[importance_df['Feature'] == feature]['Importance'].values
        feature_row[model_name] = feature_importance[0] if len(feature_importance) > 0 else 0
    comparison_data.append(feature_row)

comparison_df = pd.DataFrame(comparison_data)


for col in comparison_df.columns[1:]:
    max_val = comparison_df[col].max()
    if max_val > 0:
        comparison_df[col] = comparison_df[col] / max_val

comparison_df['Average'] = comparison_df[model_names].mean(axis=1)
comparison_df = comparison_df.sort_values('Average', ascending=False).head(20)

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    comparison_df[model_names].set_index(comparison_df['Feature']).T,
    cmap='YlOrRd',
    annot=True,
    fmt='.2f',
    cbar_kws={'label': 'Normalized Importance'}
)
plt.title('Feature Importance Comparison Across Models (Top 20 Features)')
plt.xlabel('Features')
plt.ylabel('Models')
plt.tight_layout()
plt.show()


print("\n" + "="*50)
print("TOP 10 FEATURES BY MODEL")
print("="*50)

for model_name in model_names:
    print(f"\n{model_name}:")
    print(feature_importances[model_name].head(10).to_string(index=False))


print("\n" + "="*50)
print("CONSENSUS FEATURES (Top in Multiple Models)")
print("="*50)


feature_counts = {}
for model_name, importance_df in feature_importances.items():
    top_features = importance_df.head(10)['Feature'].values
    for feature in top_features:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1


consensus_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
print("\nFeatures appearing in top 10 of multiple models:")
for feature, count in consensus_features[:15]:
    models_list = [m for m in model_names if feature in feature_importances[m].head(10)['Feature'].values]
    print(f"{feature}: {count}/5 models - [{', '.join(models_list)}]")
