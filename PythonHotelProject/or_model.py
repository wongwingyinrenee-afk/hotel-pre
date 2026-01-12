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

df = pd.read_csv('hotel_or_modeling_data.csv')


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

from sklearn.ensemble import StackingClassifier
from itertools import combinations


print("\n" + "=" * 50)
print("STACKING ENSEMBLE - TESTING ALL COMBINATIONS")
print("=" * 50)

# Define individual models for stacking
models_for_stacking = {
    'lr': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
    'dt': DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=100,
                                 class_weight='balanced', random_state=42),
    'xgb': XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=300,
                         scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                         random_state=42, eval_metric='logloss')
}

model_names_map = {
    'lr': 'Logistic Regression',
    'dt': 'Decision Tree',
    'xgb': 'XGBoost'
}

# Test all pairs of models
stacking_results = {}
all_pairs = list(combinations(models_for_stacking.keys(), 2))

print(f"\nTesting {len(all_pairs)} different stacking combinations:\n")

for pair in all_pairs:
    model1, model2 = pair
    pair_name = f"{model_names_map[model1]} + {model_names_map[model2]}"

    print(f"Training: {pair_name}...")

    # Create base estimators for this pair
    base_estimators = [
        (model1, models_for_stacking[model1]),
        (model2, models_for_stacking[model2])
    ]

    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=-1
    )

    # Train
    stacking.fit(X_train, y_train)

    # Predictions
    y_pred_stack = stacking.predict(X_test)
    y_prob_stack = stacking.predict_proba(X_test)[:, 1]

    # Store results
    stacking_results[pair_name] = {
        'accuracy': accuracy_score(y_test, y_pred_stack),
        'f1': f1_score(y_test, y_pred_stack),
        'auc': roc_auc_score(y_test, y_prob_stack),
        'model': stacking,
        'y_prob': y_prob_stack
    }

    print(f"  Accuracy: {stacking_results[pair_name]['accuracy']:.4f}")
    print(f"  F1-Score: {stacking_results[pair_name]['f1']:.4f}")
    print(f"  ROC AUC: {stacking_results[pair_name]['auc']:.4f}\n")


print("\n" + "=" * 50)
print("STACKING COMBINATIONS COMPARISON")
print("=" * 50)

stacking_comparison = pd.DataFrame({
    name: {'Accuracy': res['accuracy'], 'F1-Score': res['f1'], 'ROC AUC': res['auc']}
    for name, res in stacking_results.items()
}).T

stacking_comparison = stacking_comparison.sort_values('ROC AUC', ascending=False)
print("\n", stacking_comparison)

best_combination = stacking_comparison.index[0]
print(f"\n BEST STACKING COMBINATION: {best_combination}")
print(f"   ROC AUC: {stacking_comparison.loc[best_combination, 'ROC AUC']:.4f}")
print(f"   F1-Score: {stacking_comparison.loc[best_combination, 'F1-Score']:.4f}")
print(f"   Accuracy: {stacking_comparison.loc[best_combination, 'Accuracy']:.4f}")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, metric in enumerate(['Accuracy', 'F1-Score', 'ROC AUC']):
    stacking_comparison[metric].plot(kind='barh', ax=axes[idx], color='teal')
    axes[idx].set_title(f'{metric} - Stacking Combinations')
    axes[idx].set_xlabel(metric)
plt.tight_layout()
plt.show()


best_stack_result = stacking_results[best_combination]
results['Stacking (Best)'] = {
    'accuracy': best_stack_result['accuracy'],
    'f1': best_stack_result['f1'],
    'auc': best_stack_result['auc']
}


print(f"\nCalculating feature importance for best stacking model ({best_combination})...")
best_stack_model = best_stack_result['model']
perm_importance_stack = permutation_importance(
    best_stack_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring='roc_auc'
)
stack_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': perm_importance_stack.importances_mean
}).sort_values('Importance', ascending=False)
feature_importances['Stacking (Best)'] = stack_importance

print("\nTop 10 Most Important Features:")
print(stack_importance.head(10).to_string(index=False))


print("\n" + "=" * 50)
print("FINAL MODEL COMPARISON - INCLUDING BEST STACKING")
print("=" * 50)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('auc', ascending=False)
print(results_df)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, metric in enumerate(['accuracy', 'f1', 'auc']):
    results_df[metric].plot(kind='barh', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{metric.upper()} Comparison')
    axes[idx].set_xlabel(metric.capitalize())
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))


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
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2, linestyle='--', alpha=0.6)


for name, stack_res in stacking_results.items():
    fpr, tpr, _ = roc_curve(y_test, stack_res['y_prob'])
    auc = stack_res['auc']
    linewidth = 3 if name == best_combination else 2
    plt.plot(fpr, tpr, label=f'Stack: {name} (AUC = {auc:.3f})', linewidth=linewidth)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - All Models and Stacking Combinations')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.show()


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP', 'Stacking (Best)']
colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum', 'lightsalmon', 'gold']

for idx, model_name in enumerate(model_names):
    importance_df = feature_importances[model_name].head(15)
    axes[idx].barh(importance_df['Feature'], importance_df['Importance'], color=colors[idx])
    axes[idx].set_xlabel('Importance')
    axes[idx].set_title(f'Top 15 Features - {model_name}')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.show()


print("\n" + "=" * 50)
print("CONSENSUS FEATURES (Top in Multiple Models)")
print("=" * 50)

feature_counts = {}
for model_name, importance_df in feature_importances.items():
    top_features = importance_df.head(10)['Feature'].values
    for feature in top_features:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1

consensus_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
print("\nFeatures appearing in top 10 of multiple models:")
for feature, count in consensus_features[:15]:
    models_list = [m for m in model_names if feature in feature_importances[m].head(10)['Feature'].values]
    print(f"{feature}: {count}/6 models - [{', '.join(models_list)}]")