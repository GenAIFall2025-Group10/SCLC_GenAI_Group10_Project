"""
OncoDetect-AI: Fixed Risk Score Agent
Corrected training to learn proper risk patterns
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.connections.snowflake_connector import load_clinical_data


class FixedRiskScoreAgent:
    """Fixed Binary SCLC Risk Classification"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = None
        self.feature_names = None
        self.selected_features = None
        self.feature_importance = None
    
    def load_data(self):
        """Load training data from Snowflake"""
        print("📊 Loading data from Snowflake...")
        df = load_clinical_data()
        print(f"✓ Loaded {len(df)} samples")
        return df
    
    def create_binary_target(self, df):
        """Create risk target based on BIOLOGICAL factors, not just survival"""
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # OPTION 1: Use biological risk factors
        # HIGH RISK = High TMB OR High mutations OR Advanced stage
        df['biological_high_risk'] = (
            (df['tmb'] > df['tmb'].median()) |
            (df['mutation_count'] > df['mutation_count'].median()) |
            (df['is_advanced_stage'] == 1)
        ).astype(int)
        
        # OPTION 2: Use survival as target (original approach)
        df['survival_high_risk'] = (df['target_category'] == 'SHORT').astype(int)
        
        # HYBRID APPROACH: Combine both
        # HIGH RISK = (High biological factors AND short survival) OR (Very high biological factors)
        df['binary_risk'] = (
            ((df['tmb'] > df['tmb'].quantile(0.75)) | 
             (df['mutation_count'] > df['mutation_count'].quantile(0.75))) |
            ((df['target_category'] == 'SHORT') & 
             (df['tmb'] > df['tmb'].median()))
        ).astype(int)
        
        print("\n🎯 Binary Target Created:")
        print(f"   HIGH RISK (1): {(df['binary_risk'] == 1).sum()} samples")
        print(f"   LOW RISK (0): {(df['binary_risk'] == 0).sum()} samples")
        
        # VALIDATION
        print("\n🔍 Validating Target Logic:")
        high_risk_df = df[df['binary_risk'] == 1]
        low_risk_df = df[df['binary_risk'] == 0]
        
        print(f"\nHIGH RISK patients:")
        print(f"  Mean Age: {high_risk_df['age'].mean():.1f} years")
        print(f"  Mean TMB: {high_risk_df['tmb'].mean():.1f}")
        print(f"  Mean Mutations: {high_risk_df['mutation_count'].mean():.0f}")
        print(f"  Mean Survival: {high_risk_df['target_survival_months'].mean():.1f} months")
        
        print(f"\nLOW RISK patients:")
        print(f"  Mean Age: {low_risk_df['age'].mean():.1f} years")
        print(f"  Mean TMB: {low_risk_df['tmb'].mean():.1f}")
        print(f"  Mean Mutations: {low_risk_df['mutation_count'].mean():.0f}")
        print(f"  Mean Survival: {low_risk_df['target_survival_months'].mean():.1f} months")
        
        # Check if this makes more sense
        if high_risk_df['tmb'].mean() > low_risk_df['tmb'].mean():
            print(f"\n✓ GOOD: High risk patients have higher TMB")
        else:
            print(f"\n⚠️ WARNING: High risk patients have LOWER TMB - target may be wrong")
        
        if high_risk_df['mutation_count'].mean() > low_risk_df['mutation_count'].mean():
            print(f"✓ GOOD: High risk patients have more mutations")
        else:
            print(f"⚠️ WARNING: High risk patients have FEWER mutations - target may be wrong")
        
        return df
    
    def prepare_features(self, df, is_training=True, n_features=9):
        """Prepare features - FORCE include age + select best others"""
        print("\n🔧 Preparing features...")
        
        # Define features - age is clinically important!
        all_features = [
            'age',                    # MUST include - clinically important
            'mutation_count',         # More mutations = higher risk
            'tmb',                    # Higher TMB = higher risk
            'is_male',                # Gender differences
            'is_former_smoker',       # Smoking history
            'is_current_smoker',
            'is_never_smoker',
            'is_tmb_high',           # TMB categories
            'is_tmb_intermediate',
            'is_tmb_low',
            'is_advanced_stage',      # Stage indicators
            'age_x_advanced_stage',   # Interaction
            'smoker_x_tmb',          # Interaction
            'high_risk_flag'
        ]
        
        X = df[all_features].copy()
        
        # Feature Selection FIRST (before imputing)
        if is_training and 'binary_risk' in df.columns:
            print(f"\n🎯 Selecting top {n_features} features (including age)...")
            
            y = df['binary_risk']
            
            # Remove age from selection list temporarily
            features_for_selection = [f for f in all_features if f != 'age']
            X_for_selection = X[features_for_selection]
            
            # Select top n-1 features (save 1 spot for age)
            selector = SelectKBest(f_classif, k=n_features-1)
            selector.fit(X_for_selection.fillna(X_for_selection.median()), y)
            
            mask = selector.get_support()
            selected_without_age = [f for f, m in zip(features_for_selection, mask) if m]
            
            # Add age at the beginning
            self.selected_features = ['age'] + selected_without_age
            
            # Get all feature scores for display
            all_scores = []
            for feat in all_features:
                if feat == 'age':
                    all_scores.append(('age', 0.0, True))  # Force selected
                else:
                    idx = features_for_selection.index(feat)
                    all_scores.append((feat, selector.scores_[idx], feat in selected_without_age))
            
            feature_scores = pd.DataFrame(all_scores, columns=['feature', 'score', 'selected'])
            feature_scores = feature_scores.sort_values('score', ascending=False)
            
            print(f"\n✓ Selected features: {self.selected_features}")
            print(f"\n📊 Top 10 Feature Scores:")
            print(feature_scores.head(10).to_string(index=False))
            
            # NOW select only the chosen features
            X_selected = X[self.selected_features]
        else:
            if self.selected_features:
                X_selected = X[self.selected_features]
            else:
                X_selected = X
        
        # Impute missing values ON SELECTED FEATURES ONLY
        if is_training:
            X_imputed = self.imputer.fit_transform(X_selected)
        else:
            X_imputed = self.imputer.transform(X_selected)
        
        X_imputed = pd.DataFrame(X_imputed, columns=X_selected.columns, index=X.index)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X_imputed.columns, index=X.index)
        self.feature_names = list(X_scaled.columns)
        
        # Extract target
        if 'binary_risk' in df.columns:
            y = df['binary_risk']
            print(f"\n✓ Features shape: {X_scaled.shape}")
            print(f"✓ Binary target distribution:")
            print(f"   LOW RISK (0): {(y == 0).sum()}")
            print(f"   HIGH RISK (1): {(y == 1).sum()}")
            print(f"\n🔍 Validation:")
            print(f"   Imputer will be fitted on {X_selected.shape[1]} features")
            print(f"   Scaler will be fitted on {X_selected.shape[1]} features")
            return X_scaled, y
        else:
            return X_scaled, None
    
    def train(self, X, y):
        """Train with better regularization"""
        print("\n🤖 Training Fixed Binary Risk Score Agent...")
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"  HIGH RISK (1): {(y_train == 1).sum()}")
        print(f"  LOW RISK (0): {(y_train == 0).sum()}")
        print(f"Test set: {len(X_test)} samples")
        print(f"  HIGH RISK (1): {(y_test == 1).sum()}")
        print(f"  LOW RISK (0): {(y_test == 0).sum()}")
        
        # Use Random Forest with better regularization
        print("\n⚙️ Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=5,              # Limit depth to prevent overfitting
            min_samples_split=10,     # Require more samples to split
            min_samples_leaf=5,       # Require more samples in leaf
            max_features='sqrt',      # Use sqrt of features
            class_weight='balanced',  # Handle imbalance
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"\n✓ Training Accuracy: {train_score:.4f}")
        print(f"✓ Test Accuracy: {test_score:.4f}")
        print(f"✓ Overfitting Gap: {(train_score - test_score):.4f}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"✓ Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # ROC-AUC
        try:
            auc_score = roc_auc_score(y_test, y_proba)
            print(f"✓ ROC-AUC Score: {auc_score:.4f}")
        except:
            auc_score = None
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['LOW RISK', 'HIGH RISK']))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top Feature Importance:")
        print(self.feature_importance.head(10))
        
        # CRITICAL VALIDATION: Check if model learned correct patterns
        print("\n🔍 VALIDATING MODEL LEARNED CORRECTLY:")
        self._validate_model_learning(X_test, y_test, y_pred, y_proba)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc_score': auc_score,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
    
    def _validate_model_learning(self, X_test, y_test, y_pred, y_proba):
        """Validate that model learned sensible patterns"""
        # Check: Does the model give higher risk scores to actual high-risk patients?
        actual_high_risk_scores = y_proba[y_test == 1]
        actual_low_risk_scores = y_proba[y_test == 0]
        
        mean_high = actual_high_risk_scores.mean() if len(actual_high_risk_scores) > 0 else 0
        mean_low = actual_low_risk_scores.mean() if len(actual_low_risk_scores) > 0 else 0
        
        print(f"  Mean predicted probability for ACTUAL HIGH RISK patients: {mean_high:.3f}")
        print(f"  Mean predicted probability for ACTUAL LOW RISK patients: {mean_low:.3f}")
        
        if mean_high > mean_low:
            print(f"  ✓ CORRECT: Model gives higher scores to high-risk patients")
        else:
            print(f"  ⚠️ WARNING: Model may have learned inverted patterns!")
            print(f"     High risk patients getting LOWER scores than low risk")
    
    def predict_risk_score(self, X):
        """Predict binary risk score"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        proba = self.model.predict_proba(X)
        high_risk_proba = proba[:, 1]  # Probability of HIGH RISK (class 1)
        
        # Convert to risk score (0-100)
        risk_scores = high_risk_proba * 100
        
        predictions = self.model.predict(X)
        confidences = np.max(proba, axis=1)
        
        return {
            'risk_score': risk_scores,
            'risk_level': ['HIGH_RISK' if p == 1 else 'LOW_RISK' for p in predictions],
            'predicted_class': predictions,
            'confidence': confidences,
            'high_risk_probability': high_risk_proba,
            'low_risk_probability': proba[:, 0]
        }
    
    def save_model(self, filepath='models/risk_score_agent_binary_fixed.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'trained_at': datetime.now().isoformat(),
            'version': '4.0.0-fixed',
            'model_type': 'binary_classification',
            'note': 'Fixed model with proper pattern learning validation'
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to {filepath}")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r',
                    xticklabels=['LOW RISK', 'HIGH RISK'],
                    yticklabels=['LOW RISK', 'HIGH RISK'])
        plt.title('Fixed Risk Score Agent - Confusion Matrix')
        plt.ylabel('True Risk')
        plt.xlabel('Predicted Risk')
        
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        for i in range(2):
            for j in range(2):
                plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1%})',
                        ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix_fixed.png', dpi=300)
        print("📊 Saved: outputs/confusion_matrix_fixed.png")
    
    def plot_roc_curve(self, y_test, y_proba, auc_score):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Fixed Risk Score Agent - ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/roc_curve_fixed.png', dpi=300)
        print("📊 Saved: outputs/roc_curve_fixed.png")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        plt.figure(figsize=(10, 6))
        plt.barh(self.feature_importance['feature'], 
                self.feature_importance['importance'])
        plt.xlabel('Importance')
        plt.title('Fixed Risk Score Agent - Feature Importance')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_fixed.png', dpi=300)
        print("📊 Saved: outputs/feature_importance_fixed.png")


def main():
    """Main training with validation"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 70)
    print("🎯 OncoDetect-AI: FIXED Risk Score Agent Training")
    print("   With Pattern Learning Validation")
    print("=" * 70)
    
    agent = FixedRiskScoreAgent()
    
    # Load and validate data
    df = agent.load_data()
    df = agent.create_binary_target(df)
    
    # Prepare features (select top 9 features including age)
    X, y = agent.prepare_features(df, is_training=True, n_features=9)
    
    # Train model
    results = agent.train(X, y)
    
    # Visualizations
    agent.plot_confusion_matrix(results['y_test'], results['y_pred'])
    if results['auc_score']:
        agent.plot_roc_curve(results['y_test'], results['y_proba'], results['auc_score'])
    agent.plot_feature_importance()
    
    # Save model
    agent.save_model()
    
    print("\n" + "=" * 70)
    print("✅ Fixed Risk Score Agent Training Complete!")
    print("=" * 70)
    print("📁 Model: models/risk_score_agent_binary_fixed.pkl")
    print("📊 Visualizations: outputs/")
    print("\nTo use this model, update your API to load:")
    print("  risk_score_agent_binary_fixed.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()