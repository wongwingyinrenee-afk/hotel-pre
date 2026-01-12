import optuna
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import warnings
import random

warnings.simplefilter(action='ignore', category=FutureWarning)

# ========================================
# CONFIGURATION
# ========================================
RANDOM_SEED = 42
N_OPTUNA_TRIALS = 100
TEST_SIZE = 0.2
VAL_SIZE = 0.25

# Set all random seeds
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("HOTEL CANCELLATION PREDICTION - CONSISTENT PIPELINE")
print("=" * 70)
print(f"Random Seed: {RANDOM_SEED}")
print(f"Test Size: {TEST_SIZE}, Validation Size: {VAL_SIZE}")
print("=" * 70)

# ========================================
# STEP 1: LOAD DATA & FEATURE ENGINEERING
# ========================================
print("\n[STEP 1/5] Loading data and engineering features...")

df = pd.read_csv('hotel_or_modeling_data.csv')

# --------[ PATCH ] Safe engineered features (zero division) -----
ENGINEERED_FEATURES = {
    'price_per_night_per_guest': lambda df: np.where(
        (df['adults'] + df['children'] + df['babies']) > 0,
        df['adr'] / (df['adults'] + df['children'] + df['babies']),
        0,
    ),
    'high_price_peak': lambda df: df['adr'] * df['arrival_is_high_season'],
    'lead_time_squared': lambda df: df['lead_time'] ** 2,
    'booking_stability_score' : lambda df: np.where(
        (df['previous_bookings_not_canceled'] + df['previous_cancellations'] + 1) > 0,
        df['previous_bookings_not_canceled'] / (df['previous_bookings_not_canceled'] + df['previous_cancellations'] + 1),
        0,
    ),
    'weekend_ratio' : lambda df: np.where(
        (df['stays_in_weekend_nights'] + df['stays_in_week_nights'] + 1) > 0,
        df['stays_in_weekend_nights'] / (df['stays_in_weekend_nights'] + df['stays_in_week_nights'] + 1),
        0,
    ),
    'children_squared': lambda df: df['children'] ** 2
}
# ---------------------------------------------------------------

for feature_name, feature_func in ENGINEERED_FEATURES.items():
    df[feature_name] = feature_func(df)

# PATCH: Remove inf/NaN after features
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

print(f"✓ Created {len(ENGINEERED_FEATURES)} engineered features")
print(f" Features: {list(ENGINEERED_FEATURES.keys())}")

# ========================================
# STEP 2: OPTUNA OPTIMIZATION
# ========================================
print(f"\n[STEP 2/5] Running Optuna optimization ({N_OPTUNA_TRIALS} trials)...")

# Initial split for optimization
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

X_temp, X_test_opt, y_temp, y_test_opt = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)
X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
    X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_SEED
)

print(f" Optimization split: Train={len(X_train_opt)}, Val={len(X_val_opt)}, Test={len(X_test_opt)}")

def apply_transforms(X_data, selected_features, transforms):
    """Apply transformations consistently"""
    X_transformed = X_data[selected_features].copy()
    for f in selected_features:
        if transforms[f] == "square":
            X_transformed[f] = X_transformed[f] ** 2
        elif transforms[f] == "log":
            # PATCH: avoid negative/NaN before log
            data = X_transformed[f].copy()
            data[data < 0] = 0
            X_transformed[f] = np.log1p(data)
    # PATCH: Remove inf/NaN after transforms
    X_transformed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_transformed.fillna(0, inplace=True)
    return X_transformed

def objective(trial):
    feature_list = list(ENGINEERED_FEATURES.keys())
    selected_features = []
    for idx, f in enumerate(feature_list):
        if trial.suggest_categorical(f"feature_{idx}_use", [True, False]):
            selected_features.append(f)
    if len(selected_features) == 0:
        selected_features = feature_list[:2]  # Always at least 2

    # Choose transformations
    transforms = {}
    for idx, f in enumerate(selected_features):
        transforms[f] = trial.suggest_categorical(
            f"transform_{idx}_{f[:10]}", ["none", "square", "log"]
        )

    # Apply transformations
    X_train_transformed = apply_transforms(X_train_opt, selected_features, transforms)
    X_val_transformed = apply_transforms(X_val_opt, selected_features, transforms)

    # Extra safety for Optuna
    if np.any(np.isnan(X_train_transformed)) or np.any(np.isinf(X_train_transformed)):
        return 1.0  # Bad value

    model = XGBClassifier(
        n_estimators=200,
        max_depth=trial.suggest_int("max_depth", 3, 7),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1),
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED
    )
    model.fit(X_train_transformed, y_train_opt)

    y_pred = model.predict_proba(X_val_transformed)[:, 1]
    auc = roc_auc_score(y_val_opt, y_pred)
    return 1 - auc  # Minimize (1 - AUC)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
)
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

# Extract best parameters
best_trial = study.best_trial
feature_list = list(ENGINEERED_FEATURES.keys())
selected_features = [
    f for idx, f in enumerate(feature_list)
    if best_trial.params.get(f"feature_{idx}_use", False)
]
transforms = {
    f: best_trial.params.get(f"transform_{idx}_{f[:10]}", "none")
    for idx, f in enumerate(selected_features)
}

print(f"\n✓ Optimization complete!")
print(f" Best validation AUC: {1 - study.best_value:.4f}")
print(f" Selected engineered features: {len(selected_features)}/{len(ENGINEERED_FEATURES)}")
print(f" Features to add: {selected_features if selected_features else 'None - use original features only'}")
print(f" Best hyperparameters:")
print(f" - max_depth: {best_trial.params['max_depth']}")
print(f" - learning_rate: {best_trial.params['learning_rate']:.4f}")

# STEP 3: APPLY OPTIMIZED TRANSFORMATIONS
# ========================================
print("\n[STEP 3/5] Applying optimized transformations to full dataset...")

# Reload original data
df = pd.read_csv('hotel_or_modeling_data.csv')
for feature_name, feature_func in ENGINEERED_FEATURES.items():
    df[feature_name] = feature_func(df)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Apply optimized transformations
transforms_applied = 0
for idx, feature in enumerate(selected_features):
    transform_type = transforms[feature]

    if transform_type == 'square':
        print(f" ✓ Applying SQUARE transform to '{feature}'")
        df[feature] = df[feature] ** 2
        transforms_applied += 1
    elif transform_type == 'log':
        print(f" ✓ Applying LOG transform to '{feature}'")
        data = df[feature].copy()
        data[data < 0] = 0
        df[feature] = np.log1p(data)
        transforms_applied += 1
    else:
        print(f" • No transform for '{feature}' (keeping original)")

# PATCH: Remove inf/NaN after all transforms
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

print(f"\n✓ Applied {transforms_applied}/{len(selected_features)} transformations")

# ========================================
# STEP 4: TRAIN ALL MODELS
# ========================================
print("\n[STEP 4/5] Training all models with optimized features...")

# Final data split
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

print(f" Final split: Train={len(X_train)}, Test={len(X_test)}")
print(f" Class distribution - Train: {y_train.value_counts().to_dict()}")
print(f" Class distribution - Test: {y_test.value_counts().to_dict()}")

minmax_scaler = MinMaxScaler()
X_train_minmax = minmax_scaler.fit_transform(X_train)
X_test_minmax = minmax_scaler.transform(X_test)
standard_scaler = StandardScaler()
X_train_standard = standard_scaler.fit_transform(X_train)
X_test_standard = standard_scaler.transform(X_test)

results = {}
feature_importances = {}
models_probs = {}

print("\n [1/5] Training Logistic Regression...")
lr = LogisticRegression(
    class_weight='balanced',
    random_state=RANDOM_SEED,
    max_iter=1000
)
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
models_probs['Logistic Regression'] = y_prob_lr
print(f" Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
print(f" F1-Score: {results['Logistic Regression']['f1']:.4f}")
print(f" ROC AUC: {results['Logistic Regression']['auc']:.4f}")

print("\n [2/5] Training Decision Tree...")
dt = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=5,
    min_samples_leaf=100,
    class_weight='balanced',
    random_state=RANDOM_SEED
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
models_probs['Decision Tree'] = y_prob_dt
print(f" Accuracy: {results['Decision Tree']['accuracy']:.4f}")
print(f" F1-Score: {results['Decision Tree']['f1']:.4f}")
print(f" ROC AUC: {results['Decision Tree']['auc']:.4f}")

print("\n [3/5] Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',
    random_state=RANDOM_SEED,
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
models_probs['Random Forest'] = y_prob_rf
print(f" Accuracy: {results['Random Forest']['accuracy']:.4f}")
print(f" F1-Score: {results['Random Forest']['f1']:.4f}")
print(f" ROC AUC: {results['Random Forest']['auc']:.4f}")

print("\n [4/5] Training XGBoost...")
xgb = XGBClassifier(
    max_depth=best_trial.params['max_depth'],
    learning_rate=best_trial.params['learning_rate'],
    n_estimators=300,
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=RANDOM_SEED,
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
models_probs['XGBoost'] = y_prob_xgb
print(f" Accuracy: {results['XGBoost']['accuracy']:.4f}")
print(f" F1-Score: {results['XGBoost']['f1']:.4f}")
print(f" ROC AUC: {results['XGBoost']['auc']:.4f}")

print(f"\n Optimization vs Final Results:")
print(f" Validation AUC (with all features): {1 - study.best_value:.4f}")
print(f" Test AUC (with all features): {results['XGBoost']['auc']:.4f}")
if len(selected_features) > 0:
    print(f" Selected {len(selected_features)} engineered features were helpful!")
else:
    print(f" No engineered features selected - original features are best!")

print(f"\n Optimization Impact:")
print(f" Validation AUC (engineered features only): {1 - study.best_value:.4f}")
print(f" Test AUC (all features): {results['XGBoost']['auc']:.4f}")
print(
    f" Improvement: +{results['XGBoost']['auc'] - (1 - study.best_value):.4f} "
    f"({((results['XGBoost']['auc'] / (1 - study.best_value)) - 1) * 100:.1f}%)"
)

print("\n [5/5] Training Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=RANDOM_SEED
)
mlp.fit(X_train_standard, y_train)
y_pred_mlp = mlp.predict(X_test_standard)
y_prob_mlp = mlp.predict_proba(X_test_standard)[:, 1]
results['MLP'] = {
    'accuracy': accuracy_score(y_test, y_pred_mlp),
    'f1': f1_score(y_test, y_pred_mlp),
    'auc': roc_auc_score(y_test, y_prob_mlp)
}
print(" Calculating permutation importance...")
perm_importance = permutation_importance(
    mlp, X_test_standard, y_test,
    n_repeats=10,
    random_state=RANDOM_SEED,
    scoring='roc_auc'
)
mlp_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': perm_importance.importances_mean
}).sort_values('Importance', ascending=False)
feature_importances['MLP'] = mlp_importance
models_probs['MLP'] = y_prob_mlp
print(f" Accuracy: {results['MLP']['accuracy']:.4f}")
print(f" F1-Score: {results['MLP']['f1']:.4f}")
print(f" ROC AUC: {results['MLP']['auc']:.4f}")

# ========================================
# STEP 5: RESULTS & VISUALIZATIONS
# ========================================
print("\n[STEP 5/5] Generating results and visualizations...")

# Model Comparison Table
print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('auc', ascending=False)
print("\n" + results_df.to_string())

best_model = results_df.index[0]
print(f"\n🏆 BEST MODEL: {best_model}")
print(f" ROC AUC: {results_df.loc[best_model, 'auc']:.4f}")
print(f" F1-Score: {results_df.loc[best_model, 'f1']:.4f}")
print(f" Accuracy: {results_df.loc[best_model, 'accuracy']:.4f}")

# ----------------------------------------
# VISUALIZATION 1: Metrics Comparison
# ----------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, metric in enumerate(['accuracy', 'f1', 'auc']):
    results_df[metric].plot(kind='barh', ax=axes[idx], color='steelblue')
    axes[idx].set_title(f'{metric.upper()} Comparison', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(metric.capitalize(), fontsize=11)
    axes[idx].grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------
# VISUALIZATION 2: ROC Curves
# ----------------------------------------
plt.figure(figsize=(10, 8))
for name, y_prob in models_probs.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    linewidth = 3 if name == best_model else 2
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=linewidth)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison - All Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------
# VISUALIZATION 3: Feature Importance
# ----------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()
model_names = list(results.keys())
colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum', 'lightsalmon']

for idx, model_name in enumerate(model_names):
    importance_df = feature_importances[model_name].head(15)
    axes[idx].barh(importance_df['Feature'], importance_df['Importance'], color=colors[idx])
    axes[idx].set_xlabel('Importance', fontsize=10)
    axes[idx].set_title(f'Top 15 Features - {model_name}', fontsize=11, fontweight='bold')
    axes[idx].invert_yaxis()
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('feature_importance_individual.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------
# VISUALIZATION 4: Cross-Model Heatmap
# ----------------------------------------
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
# Normalize
for col in comparison_df.columns[1:]:
    max_val = comparison_df[col].max()
    if max_val > 0:
        comparison_df[col] = comparison_df[col] / max_val

comparison_df['Average'] = comparison_df[model_names].mean(axis=1)
comparison_df = comparison_df.sort_values('Average', ascending=False).head(20)

plt.figure(figsize=(14, 10))
sns.heatmap(
    comparison_df[model_names].set_index(comparison_df['Feature']).T,
    cmap='YlOrRd',
    annot=True,
    fmt='.2f',
    cbar_kws={'label': 'Normalized Importance'},
    linewidths=0.5
)
plt.title('Feature Importance Comparison Across Models (Top 20)',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Models', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------------------
# CONSENSUS FEATURES
# ----------------------------------------
print("\n" + "=" * 70)
print("CONSENSUS FEATURES (Important Across Multiple Models)")
print("=" * 70)

feature_counts = {}
for model_name, importance_df in feature_importances.items():
    top_features = importance_df.head(10)['Feature'].values
    for feature in top_features:
        feature_counts[feature] = feature_counts.get(feature, 0) + 1

consensus_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
print("\nFeatures appearing in top 10 of multiple models:")
for feature, count in consensus_features[:20]:
    models_list = [m for m in model_names if feature in feature_importances[m].head(10)['Feature'].values]
    print(f" {feature}: {count}/{len(model_names)} models - [{', '.join(models_list)}]")