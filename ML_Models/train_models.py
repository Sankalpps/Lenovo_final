import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


class ModelTrainer:
    """Train and evaluate multiple ML models"""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n=== TRAINING LOGISTIC REGRESSION ===")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['logistic_regression'] = model
        self._evaluate_model('logistic_regression', model)
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n=== TRAINING RANDOM FOREST ===")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        self._evaluate_model('random_forest', model)
        
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n=== TRAINING GRADIENT BOOSTING ===")
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = model
        self._evaluate_model('gradient_boosting', model)
        
    def train_xgboost(self):
        """Train XGBoost model"""
        print("\n=== TRAINING XGBOOST ===")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model
        self._evaluate_model('xgboost', model)
        
    def train_svm(self):
        """Train Support Vector Machine model"""
        print("\n=== TRAINING SVM ===")
        model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.models['svm'] = model
        self._evaluate_model('svm', model)
        
    def _evaluate_model(self, name, model):
        """Evaluate model on test set"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
    def train_all_models(self):
        """Train all models"""
        print("Starting model training...")
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_xgboost()
        self.train_svm()
        
    def print_summary(self):
        """Print model comparison summary"""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        summary_df = []
        for model_name, metrics in self.results.items():
            summary_df.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1 Score': f"{metrics['f1']:.4f}",
                'ROC AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        for row in summary_df:
            print(f"\n{row['Model'].upper()}")
            for key, value in row.items():
                if key != 'Model':
                    print(f"  {key}: {value}")
        
        # Find best model by F1 score
        best_model = max(self.results.items(), 
                         key=lambda x: x[1]['f1'])
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_model[0].upper()} (F1: {best_model[1]['f1']:.4f})")
        print(f"{'='*60}")
        
        return best_model[0]
        
    def save_models(self, output_dir='./trained_models'):
        """Save all trained models"""
        print(f"\nSaving models to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{name}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {name}")
        
        # Save results
        results_path = os.path.join(output_dir, 'model_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.results, f)
        print("  Saved: model_results.pkl")
        
    def plot_model_comparison(self, output_dir='./trained_models'):
        """Plot model performance comparison"""
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            values = [self.results[model][metric] for model in model_names]
            colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
            
            bars = ax.bar(model_names, values, color=colors)
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim([0, 1])
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: {os.path.join(output_dir, 'model_comparison.png')}")
        
    def plot_confusion_matrices(self, output_dir='./trained_models'):
        """Plot confusion matrices for all models"""
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Confusion Matrices', fontsize=16, fontweight='bold')
        
        model_names = list(self.results.keys())
        
        for idx, name in enumerate(model_names):
            ax = axes[idx // 3, idx % 3]
            y_pred = self.results[name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar=False, annot_kws={'size': 12})
            ax.set_title(name.replace('_', ' ').title())
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrices plot: {os.path.join(output_dir, 'confusion_matrices.png')}")


def main():
    """Main training pipeline"""
    
    # Load preprocessed data
    data_dir = './processed_data'
    
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")
    
    # Train models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    trainer.train_all_models()
    
    # Print summary and save
    best_model = trainer.print_summary()
    trainer.save_models()
    trainer.plot_model_comparison()
    trainer.plot_confusion_matrices()
    
    print("\n✓ Model training complete!")
    print(f"✓ Best model: {best_model}")
    print(f"✓ Models saved to ./trained_models/")


if __name__ == "__main__":
    main()
