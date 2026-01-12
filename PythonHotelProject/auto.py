import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
from itertools import combinations
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def run_all_models(csv_path, test_size=0.2, random_state=42):
    print("\n" + "=" * 50)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 50)
    df = pd.read_csv(csv_path)

    X = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")

    minmax_scaler = MinMaxScaler()
    X_train_minmax = minmax_scaler.fit_transform(X_train)
    X_test_minmax = minmax_scaler.transform(X_test)

    standard_scaler = StandardScaler()
    X_train_standard = standard_scaler.fit_transform(X_train)
    X_test_standard = standard_scaler.transform(X_test)

    results = {}
    feature_importances = {}
    trained_models = {}
    all_predictions = {}

    available_models = {
        '1': 'Logistic Regression',
        '2': 'Decision Tree',
        '3': 'Random Forest',
        '4': 'XGBoost',
        '5': 'Neural Network (MLP)',
        '6': 'Stacking Ensemble (All Combinations)'
    }

    while True:

        print("AVAILABLE MODELS")

        for key, name in available_models.items():
            status = "COMPLETED" if name in results or name.startswith(
                'Stacking') and 'Stacking (Best)' in results else ""
            print(f"{key}. {name} {status}")
        print("0. Exit and show summary")

        choice = input("\nWhich model do you want to run? (Enter number or 0 to exit): ").strip()

        if choice == '0':
            break

        if choice not in available_models:
            print(" Invalid choice! Please enter a number from the list.")
            continue

        model_name = available_models[choice]

        if model_name in results or (model_name.startswith('Stacking') and 'Stacking (Best)' in results):
            print(f"\n️  {model_name} has already been run!")
            rerun = input("Do you want to run it again? (yes/no): ").strip().lower()
            if rerun not in ['yes', 'y']:
                continue

        if choice == '1':  # Logistic Regression
            print("\n" + "=" * 50)
            print("RUNNING: LOGISTIC REGRESSION")
            print("=" * 50)

            lr = LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000)
            lr.fit(X_train_minmax, y_train)

            y_pred = lr.predict(X_test_minmax)
            y_prob = lr.predict_proba(X_test_minmax)[:, 1]

            results['Logistic Regression'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }

            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': np.abs(lr.coef_[0])
            }).sort_values('Importance', ascending=False)
            feature_importances['Logistic Regression'] = importance
            trained_models['Logistic Regression'] = lr
            all_predictions['Logistic Regression'] = {'pred': y_pred, 'prob': y_prob}

            print(f"\n Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
            print(f" F1-Score: {results['Logistic Regression']['f1']:.4f}")
            print(f" ROC AUC: {results['Logistic Regression']['auc']:.4f}")
            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        elif choice == '2':  # Decision Tree
            print("\n" + "=" * 50)
            print("RUNNING: DECISION TREE")
            print("=" * 50)

            dt = DecisionTreeClassifier(
                criterion='entropy',
                max_depth=5,
                min_samples_leaf=100,
                class_weight='balanced',
                random_state=random_state
            )
            dt.fit(X_train, y_train)

            y_pred = dt.predict(X_test)
            y_prob = dt.predict_proba(X_test)[:, 1]

            results['Decision Tree'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }

            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': dt.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_importances['Decision Tree'] = importance
            trained_models['Decision Tree'] = dt
            all_predictions['Decision Tree'] = {'pred': y_pred, 'prob': y_prob}

            print(f"\n Accuracy: {results['Decision Tree']['accuracy']:.4f}")
            print(f" F1-Score: {results['Decision Tree']['f1']:.4f}")
            print(f" ROC AUC: {results['Decision Tree']['auc']:.4f}")
            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        elif choice == '3':  # Random Forest
            print("\n" + "=" * 50)
            print("RUNNING: RANDOM FOREST")
            print("=" * 50)

            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_test)
            y_prob = rf.predict_proba(X_test)[:, 1]

            results['Random Forest'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }

            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_importances['Random Forest'] = importance
            trained_models['Random Forest'] = rf
            all_predictions['Random Forest'] = {'pred': y_pred, 'prob': y_prob}

            print(f"\n Accuracy: {results['Random Forest']['accuracy']:.4f}")
            print(f" F1-Score: {results['Random Forest']['f1']:.4f}")
            print(f" ROC AUC: {results['Random Forest']['auc']:.4f}")
            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        elif choice == '4':  # XGBoost
            print("\n" + "=" * 50)
            print("RUNNING: XGBOOST")
            print("=" * 50)

            xgb = XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                n_estimators=300,
                scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                random_state=random_state,
                eval_metric='logloss'
            )
            xgb.fit(X_train, y_train)

            y_pred = xgb.predict(X_test)
            y_prob = xgb.predict_proba(X_test)[:, 1]

            results['XGBoost'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }

            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': xgb.feature_importances_
            }).sort_values('Importance', ascending=False)
            feature_importances['XGBoost'] = importance
            trained_models['XGBoost'] = xgb
            all_predictions['XGBoost'] = {'pred': y_pred, 'prob': y_prob}

            print(f"\n Accuracy: {results['XGBoost']['accuracy']:.4f}")
            print(f" F1-Score: {results['XGBoost']['f1']:.4f}")
            print(f" ROC AUC: {results['XGBoost']['auc']:.4f}")
            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        elif choice == '5':  # MLP
            print("\n" + "=" * 50)
            print("RUNNING: NEURAL NETWORK (MLP)")
            print("=" * 50)

            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=random_state
            )
            mlp.fit(X_train_standard, y_train)

            y_pred = mlp.predict(X_test_standard)
            y_prob = mlp.predict_proba(X_test_standard)[:, 1]

            results['MLP'] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }

            print("Calculating permutation importance for MLP...")
            perm_importance = permutation_importance(
                mlp, X_test_standard, y_test,
                n_repeats=10,
                random_state=random_state,
                scoring='roc_auc'
            )
            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': perm_importance.importances_mean
            }).sort_values('Importance', ascending=False)
            feature_importances['MLP'] = importance
            trained_models['MLP'] = mlp
            all_predictions['MLP'] = {'pred': y_pred, 'prob': y_prob}

            print(f"\n Accuracy: {results['MLP']['accuracy']:.4f}")
            print(f" F1-Score: {results['MLP']['f1']:.4f}")
            print(f" ROC AUC: {results['MLP']['auc']:.4f}")
            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        elif choice == '6':  # Stacking Ensemble
            print("\n" + "=" * 50)
            print("RUNNING: STACKING ENSEMBLE")
            print("=" * 50)

            models_for_stacking = {
                'lr': LogisticRegression(class_weight='balanced', random_state=random_state, max_iter=1000),
                'dt': DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_leaf=100,
                                             class_weight='balanced', random_state=random_state),
                'xgb': XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=300,
                                     scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                                     random_state=random_state, eval_metric='logloss')
            }

            model_names_map = {
                'lr': 'Logistic Regression',
                'dt': 'Decision Tree',
                'xgb': 'XGBoost'
            }

            stacking_results = {}
            all_pairs = list(combinations(models_for_stacking.keys(), 2))

            print(f"\nTesting {len(all_pairs)} different stacking combinations...\n")

            for pair in all_pairs:
                model1, model2 = pair
                pair_name = f"{model_names_map[model1]} + {model_names_map[model2]}"

                print(f"Training: {pair_name}...")

                base_estimators = [
                    (model1, models_for_stacking[model1]),
                    (model2, models_for_stacking[model2])
                ]

                stacking = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=LogisticRegression(random_state=random_state, max_iter=1000),
                    cv=5,
                    n_jobs=-1
                )

                stacking.fit(X_train, y_train)

                y_pred_stack = stacking.predict(X_test)
                y_prob_stack = stacking.predict_proba(X_test)[:, 1]

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

            # Find best combination
            stacking_comparison = pd.DataFrame({
                name: {'Accuracy': res['accuracy'], 'F1-Score': res['f1'], 'ROC AUC': res['auc']}
                for name, res in stacking_results.items()
            }).T

            stacking_comparison = stacking_comparison.sort_values('ROC AUC', ascending=False)
            print("\n" + "=" * 50)
            print("STACKING COMBINATIONS COMPARISON")
            print("=" * 50)
            print(stacking_comparison)

            best_combination = stacking_comparison.index[0]
            print(f"\n🏆 BEST STACKING COMBINATION: {best_combination}")
            print(f"   ROC AUC: {stacking_comparison.loc[best_combination, 'ROC AUC']:.4f}")
            print(f"   F1-Score: {stacking_comparison.loc[best_combination, 'F1-Score']:.4f}")
            print(f"   Accuracy: {stacking_comparison.loc[best_combination, 'Accuracy']:.4f}")

            # Visualize
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for idx, metric in enumerate(['Accuracy', 'F1-Score', 'ROC AUC']):
                stacking_comparison[metric].plot(kind='barh', ax=axes[idx], color='teal')
                axes[idx].set_title(f'{metric} - Stacking Combinations')
                axes[idx].set_xlabel(metric)
            plt.tight_layout()
            plt.show()

            # Add best to results
            best_stack_result = stacking_results[best_combination]
            results['Stacking (Best)'] = {
                'accuracy': best_stack_result['accuracy'],
                'f1': best_stack_result['f1'],
                'auc': best_stack_result['auc']
            }

            print(f"\nCalculating feature importance for best stacking model...")
            best_stack_model = best_stack_result['model']
            perm_importance_stack = permutation_importance(
                best_stack_model, X_test, y_test,
                n_repeats=10,
                random_state=random_state,
                scoring='roc_auc'
            )
            importance = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': perm_importance_stack.importances_mean
            }).sort_values('Importance', ascending=False)
            feature_importances['Stacking (Best)'] = importance
            trained_models['Stacking (Best)'] = best_stack_model

            print("\nTop 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))

        print("\n Model training completed!")

        # Ask if want to continue
        cont = input("\nDo you want to run another model? (yes/no): ").strip().lower()
        if cont not in ['yes', 'y']:
            break

    if len(results) > 0:
        print("\n" + "=" * 50)
        print("FINAL SUMMARY - ALL MODELS RUN")
        print("=" * 50)

        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('auc', ascending=False)
        print(results_df)

        # Visualizations
        print("\nGenerating comparison plots...")

        # Metrics comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for idx, metric in enumerate(['accuracy', 'f1', 'auc']):
            results_df[metric].plot(kind='barh', ax=axes[idx], color='steelblue')
            axes[idx].set_title(f'{metric.upper()} Comparison')
            axes[idx].set_xlabel(metric.capitalize())
        plt.tight_layout()
        plt.show()

        # ROC Curves
        if len(all_predictions) > 0:
            plt.figure(figsize=(12, 8))

            for name, preds in all_predictions.items():
                fpr, tpr, _ = roc_curve(y_test, preds['prob'])
                auc = roc_auc_score(y_test, preds['prob'])
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.3)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - All Models')
            plt.legend(loc='lower right')
            plt.grid(alpha=0.3)
            plt.show()

        # Feature Importance
        if len(feature_importances) > 0:
            n_models = len(feature_importances)
            n_cols = 3
            n_rows = (n_models + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            axes = axes.flatten()

            colors = ['skyblue', 'lightgreen', 'lightcoral', 'plum', 'lightsalmon', 'gold']

            for idx, (model_name, importance_df) in enumerate(feature_importances.items()):
                importance_plot = importance_df.head(15)
                axes[idx].barh(importance_plot['Feature'], importance_plot['Importance'],
                               color=colors[idx % len(colors)])
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'Top 15 Features - {model_name}')
                axes[idx].invert_yaxis()

            # Hide empty subplots
            for idx in range(len(feature_importances), len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.show()

        # Consensus Features
        if len(feature_importances) >= 2:
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
                models_list = [m for m in feature_importances.keys()
                               if feature in feature_importances[m].head(10)['Feature'].values]
                print(f"{feature}: {count}/{len(feature_importances)} models - [{', '.join(models_list)}]")

        print("\n" + "=" * 50)
        print("=" * 50)

        return {
            'results': results_df,
            'feature_importances': feature_importances,
            'models': trained_models,
            'predictions': all_predictions,
            'test_data': (X_test, y_test)
        }
    else:
        print("\n No models were run!")
        return None
