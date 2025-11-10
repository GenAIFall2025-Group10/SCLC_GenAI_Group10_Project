"""
OncoDetect-AI: SCLC Subtype Classifier Agent
Predicts SCLC molecular subtype (SCLC-A, SCLC-N, SCLC-P, SCLC-Y)
Based on biomarker gene expression
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Snowflake connection
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.connections.snowflake_connector import SnowflakeConnector, OncoDetectQueries

import sklearn
sklearn.set_config(transform_output="default")


class SubtypeClassifierAgent:
    """SCLC Subtype Classification Agent"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.feature_importance = None
        self.classes = None
    
    def load_data_from_snowflake(self):
        """Load genomic features from Snowflake"""
        print("📊 Loading genomic data from Snowflake...")
        
        query = """
        SELECT 
            sample_id,
            sclc_subtype,
            ascl1_expression,
            neurod1_expression,
            pou2f3_expression,
            yap1_expression,
            tp53_expression,
            rb1_expression,
            myc_expression,
            mycl_expression,
            mycn_expression,
            dll3_expression,
            bcl2_expression,
            notch1_expression,
            myc_family_score,
            tp53_rb1_dual_loss,
            mean_expression,
            stddev_expression,
            genes_expressed
        FROM MART_RISK_FEATURES_GENOMIC
        WHERE sclc_subtype IS NOT NULL
          AND sclc_subtype != 'MIXED'
        ORDER BY sample_id
        """
        
        with SnowflakeConnector() as sf:
            df = sf.execute_query(query)
        
        df.columns = df.columns.str.lower()
        print(f"✓ Loaded {len(df)} samples")
        
        # Show subtype distribution
        print(f"\n🧬 Subtype Distribution:")
        print(df['sclc_subtype'].value_counts())
        
        return df
    
    def prepare_features(self, df, is_training=True):
        """Prepare biomarker features for subtype classification"""
        print("\n🔧 Preparing biomarker features...")
        
        # Key biomarker features for subtype classification
        biomarker_features = [
            'ascl1_expression',
            'neurod1_expression',
            'pou2f3_expression',
            'yap1_expression',
            'tp53_expression',
            'rb1_expression',
            'myc_family_score',
            'tp53_rb1_dual_loss',
            'dll3_expression',
            'bcl2_expression',
            'notch1_expression',
            'mean_expression',
            'stddev_expression',
            'genes_expressed'
        ]
        
        # Extract features as numpy array
        X = df[biomarker_features].values
        
        # Impute missing values
        if is_training:
            X_imputed = self.imputer.fit_transform(X)
        else:
            X_imputed = self.imputer.transform(X)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = self.scaler.transform(X_imputed)
        
        self.feature_names = biomarker_features
        
        # Convert back to DataFrame for convenience
        X_final = pd.DataFrame(X_scaled, columns=biomarker_features, index=df.index)
        
        # Extract target (subtype)
        if 'sclc_subtype' in df.columns:
            y = df['sclc_subtype']
            print(f"✓ Features shape: {X_final.shape}")
            print(f"✓ Subtype distribution:")
            print(y.value_counts())
            return X_final, y
        else:
            return X_final, None
    
    def train(self, X, y):
        """Train SCLC subtype classifier"""
        print("\n🤖 Training SCLC Subtype Classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest (good for subtype classification)
        print("\n⚙️ Training Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train.values, y_train.values)
        
        # Store classes
        self.classes = self.model.classes_
        
        # Evaluate
        train_score = self.model.score(X_train.values, y_train.values)
        test_score = self.model.score(X_test.values, y_test.values)
        
        print(f"\n✓ Training Accuracy: {train_score:.4f}")
        print(f"✓ Test Accuracy: {test_score:.4f}")
        print(f"✓ Overfitting Gap: {(train_score - test_score):.4f}")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train.values, y_train.values, cv=cv)
        print(f"✓ Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Predictions
        y_pred = self.model.predict(X_test.values)
        
        # Classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔝 Top 10 Most Important Features:")
        print(self.feature_importance.head(10))
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
    
    def predict_subtype(self, X):
        """Predict SCLC subtype with probabilities"""
        if self.model is None:
            raise ValueError("Model not trained!")
        
        # Predict
        predictions = self.model.predict(X.values)
        probabilities = self.model.predict_proba(X.values)
        
        # Get confidence (max probability)
        confidences = np.max(probabilities, axis=1)
        
        results = []
        for i in range(len(predictions)):
            result = {
                'predicted_subtype': predictions[i],
                'confidence': confidences[i],
                'probabilities': {
                    subtype: float(prob) 
                    for subtype, prob in zip(self.classes, probabilities[i])
                }
            }
            results.append(result)
        
        return results
    
    def save_model(self, filepath='models/subtype_classifier.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'classes': self.classes,
            'trained_at': datetime.now().isoformat(),
            'version': '1.0.0-subtype',
            'model_type': 'subtype_classification'
        }
        
        joblib.dump(model_data, filepath)
        print(f"\n💾 Model saved to {filepath}")
    
    def load_model(self, filepath='models/subtype_classifier.pkl'):
        """Load pre-trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.classes = model_data['classes']
        
        print(f"✓ Model loaded from {filepath}")
        print(f"✓ Version: {model_data.get('version', 'unknown')}")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix for subtypes"""
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        plt.title('SCLC Subtype Classifier - Confusion Matrix')
        plt.ylabel('True Subtype')
        plt.xlabel('Predicted Subtype')
        plt.tight_layout()
        plt.savefig('outputs/confusion_matrix_subtype.png', dpi=300)
        print("\n📊 Confusion matrix saved to outputs/confusion_matrix_subtype.png")
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(14)
        colors = sns.color_palette('viridis', len(top_features))
        plt.barh(top_features['feature'], top_features['importance'], color=colors)
        plt.xlabel('Importance')
        plt.title('SCLC Subtype Classifier - Feature Importance')
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_subtype.png', dpi=300)
        print("📊 Feature importance saved to outputs/feature_importance_subtype.png")
    
    def plot_subtype_distribution(self, y_test, y_pred):
        """Plot subtype distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual distribution
        actual_counts = pd.Series(y_test).value_counts()
        axes[0].bar(actual_counts.index, actual_counts.values, color='steelblue', alpha=0.7)
        axes[0].set_title('Actual Subtype Distribution (Test Set)')
        axes[0].set_xlabel('SCLC Subtype')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Predicted distribution
        pred_counts = pd.Series(y_pred).value_counts()
        axes[1].bar(pred_counts.index, pred_counts.values, color='coral', alpha=0.7)
        axes[1].set_title('Predicted Subtype Distribution (Test Set)')
        axes[1].set_xlabel('SCLC Subtype')
        axes[1].set_ylabel('Count')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/subtype_distribution.png', dpi=300)
        print("📊 Subtype distribution saved to outputs/subtype_distribution.png")


def main():
    """Main training execution"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    print("=" * 70)
    print("🧬 OncoDetect-AI: SCLC Subtype Classifier Training")
    print("=" * 70)
    
    # Initialize agent
    agent = SubtypeClassifierAgent()
    
    # Load genomic data from Snowflake
    df = agent.load_data_from_snowflake()
    
    # Prepare features
    X, y = agent.prepare_features(df, is_training=True)
    
    # Train model
    results = agent.train(X, y)
    
    # Visualizations
    agent.plot_confusion_matrix(results['y_test'], results['y_pred'])
    agent.plot_feature_importance()
    agent.plot_subtype_distribution(results['y_test'], results['y_pred'])
    
    # Save model
    agent.save_model()
    
    # Test predictions on sample patients
    print("\n" + "=" * 70)
    print("🔮 Testing Subtype Predictions")
    print("=" * 70)
    
    test_samples = X.sample(min(5, len(X)))
    predictions = agent.predict_subtype(test_samples)
    
    # Get actual subtypes for comparison
    actual_data = df.loc[test_samples.index]
    
    for i, idx in enumerate(test_samples.index):
        actual_subtype = actual_data.loc[idx, 'sclc_subtype']
        pred = predictions[i]
        
        print(f"\n{'='*60}")
        print(f"Sample: {idx}")
        print(f"{'='*60}")
        
        # Show key biomarkers
        ascl1 = actual_data.loc[idx, 'ascl1_expression']
        neurod1 = actual_data.loc[idx, 'neurod1_expression']
        pou2f3 = actual_data.loc[idx, 'pou2f3_expression']
        yap1 = actual_data.loc[idx, 'yap1_expression']
        
        print(f"Biomarker Profile:")
        print(f"  • ASCL1: {ascl1:.2f}")
        print(f"  • NEUROD1: {neurod1:.2f}")
        print(f"  • POU2F3: {pou2f3:.2f}")
        print(f"  • YAP1: {yap1:.2f}")
        
        print(f"\nActual Subtype: {actual_subtype}")
        print(f"Predicted Subtype: {pred['predicted_subtype']}")
        print(f"Confidence: {pred['confidence']:.1%}")
        
        print(f"\nSubtype Probabilities:")
        for subtype, prob in pred['probabilities'].items():
            print(f"  • {subtype}: {prob:.1%}")
        
        print(f"\nResult: {'✓ CORRECT' if pred['predicted_subtype'] == actual_subtype else '✗ INCORRECT'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 Model Performance Summary")
    print("=" * 70)
    print(f"Test Accuracy: {results['test_accuracy']:.1%}")
    print(f"Cross-Validation: {results['cv_mean']:.1%} (+/- {results['cv_std']:.1%})")
    print(f"Training Samples: {len(X)}")
    print(f"SCLC Subtypes: {list(agent.classes)}")
    
    print("\n" + "=" * 70)
    print("✅ SCLC Subtype Classifier Training Complete!")
    print("=" * 70)
    print(f"📁 Model: models/subtype_classifier.pkl")
    print(f"📊 Visualizations: outputs/")
    print(f"\n🎯 Clinical Use:")
    print(f"   • SCLC-A: Classic neuroendocrine (DLL3 targetable)")
    print(f"   • SCLC-N: Alternative neuroendocrine")
    print(f"   • SCLC-P: Tuft-cell like variant")
    print(f"   • SCLC-Y: Non-neuroendocrine (YAP1 driven)")
    print("=" * 70)


if __name__ == "__main__":
    main()