import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from io import BytesIO
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class MLPipeline:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.label_encoders = {}
        
    def set_data(self, data):
        """Set the data for ML pipeline"""
        self.data = data
    
    def auto_ml_pipeline(self, target_col, problem_type='auto', test_size=0.2, cv_folds=5):
        """Complete automated ML pipeline"""
        if self.data is None or target_col not in self.data.columns:
            st.error("Data not available or target column not found!")
            return None
        
        try:
            # Prepare data
            self._prepare_data(target_col)
            
            # Check if we have enough data
            if len(self.X) < 10:
                st.error("Not enough data for machine learning (minimum 10 rows required)")
                return None
            
            # Determine problem type
            if problem_type == 'auto':
                problem_type = self._detect_problem_type()
            
            # Check if target is appropriate for the problem type
            if problem_type == 'classification' and self.y.nunique() > 50:
                st.warning(f"Target column has {self.y.nunique()} unique values. This might be too many for classification.")
                st.info("Consider selecting a different target column or using regression instead.")
                return None
            
            # Split data
            try:
                if problem_type == 'classification' and len(self.y.unique()) > 1:
                    # Stratify only if we have multiple classes
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                        self.X, self.y, test_size=test_size, random_state=42, stratify=self.y
                    )
                else:
                    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                        self.X, self.y, test_size=test_size, random_state=42
                    )
            except ValueError as e:
                # Fallback to non-stratified split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    self.X, self.y, test_size=test_size, random_state=42
                )
            
            # Create preprocessor
            self._create_preprocessor()
            
            # Train models
            if problem_type == 'classification':
                models_to_train = self._get_classification_models()
            else:
                models_to_train = self._get_regression_models()
            
            results = self._train_models(models_to_train, problem_type, cv_folds)
            
            if not results:
                st.error("No models could be trained successfully. Please check your data.")
                return None
            
            # Select best model
            self._select_best_model(results, problem_type)
            
            return {
                'problem_type': problem_type,
                'results': results,
                'best_model': self.best_model_name,
                'feature_names': list(self.X.columns),
                'target_column': target_col,
                'test_size': test_size,
                'data_shape': self.X.shape
            }
            
        except Exception as e:
            st.error(f"Error in ML pipeline: {str(e)}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")
            return None
    
    def _prepare_data(self, target_col):
        """Prepare data for ML"""
        # Separate features and target
        self.X = self.data.drop(columns=[target_col])
        self.y = self.data[target_col]
        
        # Handle missing values in features
        for col in self.X.columns:
            if self.X[col].dtype in ['object', 'category']:
                # Convert categorical to object first to allow new categories
                if self.X[col].dtype == 'category':
                    self.X[col] = self.X[col].astype('object')
                self.X[col] = self.X[col].fillna('Unknown')
            else:
                self.X[col] = self.X[col].fillna(self.X[col].median())
        
        # Handle missing values in target
        if self.y.isnull().any():
            mask = ~self.y.isnull()
            self.X = self.X[mask]
            self.y = self.y[mask]
            
        # Convert target to object if it's categorical to avoid category issues
        if self.y.dtype == 'category':
            self.y = self.y.astype('object')
    
    def _detect_problem_type(self):
        """Automatically detect if it's classification or regression"""
        if self.y.dtype == 'object' or self.y.dtype == 'category':
            return 'classification'
        elif self.y.nunique() <= 20 and self.y.nunique() / len(self.y) < 0.1:
            return 'classification'
        else:
            return 'regression'
    
    def _create_preprocessor(self):
        """Create preprocessing pipeline"""
        numeric_features = self.X.select_dtypes(include=[np.number]).columns
        categorical_features = self.X.select_dtypes(include=['object', 'category']).columns
        
        # Numeric transformer
        numeric_transformer = StandardScaler()
        
        # Categorical transformer
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
    
    def _get_classification_models(self):
        """Get classification models to train"""
        return {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Machine': SVC(random_state=42, probability=True)
        }
    
    def _get_regression_models(self):
        """Get regression models to train"""
        return {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Support Vector Machine': SVR()
        }
    
    def _train_models(self, models_to_train, problem_type, cv_folds):
        """Train multiple models and evaluate them"""
        results = {}
        
        for name, model in models_to_train.items():
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier' if problem_type == 'classification' else 'regressor', model)
                ])
                
                # Fit the pipeline
                pipeline.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = pipeline.predict(self.X_test)
                y_train_pred = pipeline.predict(self.X_train)
                
                # Calculate metrics
                if problem_type == 'classification':
                    # Handle target encoding for string labels
                    if self.y.dtype == 'object':
                        if not hasattr(self, 'target_encoder'):
                            self.target_encoder = LabelEncoder()
                            y_train_encoded = self.target_encoder.fit_transform(self.y_train)
                            y_test_encoded = self.target_encoder.transform(self.y_test)
                            y_pred_encoded = self.target_encoder.transform(y_pred)
                        else:
                            y_train_encoded = self.target_encoder.transform(self.y_train)
                            y_test_encoded = self.target_encoder.transform(self.y_test)
                            y_pred_encoded = self.target_encoder.transform(y_pred)
                    else:
                        y_train_encoded = self.y_train
                        y_test_encoded = self.y_test
                        y_pred_encoded = y_pred
                    
                    test_accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
                    train_accuracy = accuracy_score(y_train_encoded, self.target_encoder.transform(y_train_pred) if self.y.dtype == 'object' else y_train_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv_folds, scoring='accuracy')
                    
                    results[name] = {
                        'model': pipeline,
                        'test_accuracy': test_accuracy,
                        'train_accuracy': train_accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'overfitting': abs(train_accuracy - test_accuracy)
                    }
                
                else:  # regression
                    test_r2 = r2_score(self.y_test, y_pred)
                    train_r2 = r2_score(self.y_train, y_train_pred)
                    test_mse = mean_squared_error(self.y_test, y_pred)
                    train_mse = mean_squared_error(self.y_train, y_train_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv_folds, scoring='r2')
                    
                    results[name] = {
                        'model': pipeline,
                        'test_r2': test_r2,
                        'train_r2': train_r2,
                        'test_mse': test_mse,
                        'train_mse': train_mse,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'overfitting': abs(train_r2 - test_r2)
                    }
                
            except Exception as e:
                st.warning(f"Failed to train {name}: {e}")
                continue
        
        return results
    
    def _select_best_model(self, results, problem_type):
        """Select the best performing model"""
        if not results:
            return
        
        if problem_type == 'classification':
            # Select based on test accuracy with penalty for overfitting
            best_score = -1
            for name, metrics in results.items():
                score = metrics['test_accuracy'] - (metrics['overfitting'] * 0.1)  # Penalty for overfitting
                if score > best_score:
                    best_score = score
                    self.best_model_name = name
                    self.best_model = metrics['model']
        else:
            # Select based on test R2 with penalty for overfitting
            best_score = -1
            for name, metrics in results.items():
                score = metrics['test_r2'] - (metrics['overfitting'] * 0.1)  # Penalty for overfitting
                if score > best_score:
                    best_score = score
                    self.best_model_name = name
                    self.best_model = metrics['model']
    
    def predict(self, new_data):
        """Make predictions on new data"""
        if self.best_model is None:
            st.error("No trained model available!")
            return None
        
        try:
            predictions = self.best_model.predict(new_data)
            return predictions
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def get_feature_importance(self):
        """Get feature importance from the best model"""
        if self.best_model is None:
            return None
        
        try:
            # Get the actual model from the pipeline
            model = self.best_model.named_steps['classifier'] if 'classifier' in self.best_model.named_steps else self.best_model.named_steps['regressor']
            
            # Check if model has feature importance
            if hasattr(model, 'feature_importances_'):
                # Get feature names after preprocessing
                preprocessor = self.best_model.named_steps['preprocessor']
                
                # Get feature names from preprocessor
                try:
                    feature_names = []
                    
                    # Numeric features
                    numeric_features = preprocessor.named_transformers_['num'].get_feature_names_out() if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out') else self.X.select_dtypes(include=[np.number]).columns
                    feature_names.extend(numeric_features)
                    
                    # Categorical features (one-hot encoded)
                    if 'cat' in preprocessor.named_transformers_:
                        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out() if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out') else []
                        feature_names.extend(cat_features)
                    
                except:
                    # Fallback to original feature names
                    feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names[:len(model.feature_importances_)],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
            
            elif hasattr(model, 'coef_'):
                # For linear models
                feature_names = self.X.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': abs(model.coef_[0]) if model.coef_.ndim > 1 else abs(model.coef_)
                }).sort_values('importance', ascending=False)
                
                return importance_df
            
        except Exception as e:
            st.warning(f"Could not extract feature importance: {e}")
            return None
    
    def create_model_comparison_plot(self, results, problem_type):
        """Create a plot comparing model performance"""
        if not results:
            return None
        
        model_names = list(results.keys())
        
        if problem_type == 'classification':
            test_scores = [results[name]['test_accuracy'] for name in model_names]
            cv_scores = [results[name]['cv_mean'] for name in model_names]
            cv_stds = [results[name]['cv_std'] for name in model_names]
            
            fig = go.Figure()
            
            # Test accuracy
            fig.add_trace(go.Bar(
                name='Test Accuracy',
                x=model_names,
                y=test_scores,
                marker_color='lightblue'
            ))
            
            # CV accuracy with error bars
            fig.add_trace(go.Bar(
                name='CV Mean Accuracy',
                x=model_names,
                y=cv_scores,
                error_y=dict(type='data', array=cv_stds),
                marker_color='darkblue'
            ))
            
            fig.update_layout(
                title='Model Performance Comparison (Accuracy)',
                xaxis_title='Models',
                yaxis_title='Accuracy',
                barmode='group',
                template='plotly_white'
            )
        
        else:  # regression
            test_scores = [results[name]['test_r2'] for name in model_names]
            cv_scores = [results[name]['cv_mean'] for name in model_names]
            cv_stds = [results[name]['cv_std'] for name in model_names]
            
            fig = go.Figure()
            
            # Test R2
            fig.add_trace(go.Bar(
                name='Test R²',
                x=model_names,
                y=test_scores,
                marker_color='lightgreen'
            ))
            
            # CV R2 with error bars
            fig.add_trace(go.Bar(
                name='CV Mean R²',
                x=model_names,
                y=cv_scores,
                error_y=dict(type='data', array=cv_stds),
                marker_color='darkgreen'
            ))
            
            fig.update_layout(
                title='Model Performance Comparison (R² Score)',
                xaxis_title='Models',
                yaxis_title='R² Score',
                barmode='group',
                template='plotly_white'
            )
        
        return fig
    
    def create_prediction_plot(self, results, problem_type):
        """Create prediction vs actual plot for the best model"""
        if self.best_model is None or self.best_model_name not in results:
            return None
        
        predictions = results[self.best_model_name]['predictions']
        
        if problem_type == 'regression':
            fig = px.scatter(
                x=self.y_test,
                y=predictions,
                title=f'Predictions vs Actual ({self.best_model_name})',
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                template='plotly_white'
            )
            
            # Add diagonal line for perfect predictions
            min_val = min(min(self.y_test), min(predictions))
            max_val = max(max(self.y_test), max(predictions))
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='red')
            ))
            
        else:  # classification
            # Confusion matrix visualization
            if self.y.dtype == 'object' and hasattr(self, 'target_encoder'):
                y_test_encoded = self.target_encoder.transform(self.y_test)
                predictions_encoded = self.target_encoder.transform(predictions)
                labels = self.target_encoder.classes_
            else:
                y_test_encoded = self.y_test
                predictions_encoded = predictions
                labels = sorted(self.y.unique())
            
            cm = confusion_matrix(y_test_encoded, predictions_encoded)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                x=labels,
                y=labels,
                title=f'Confusion Matrix ({self.best_model_name})',
                template='plotly_white'
            )
            fig.update_layout(
                xaxis_title='Predicted',
                yaxis_title='Actual'
            )
        
        return fig
    
    def export_model(self):
        """Export the best trained model"""
        if self.best_model is None:
            return None
        
        buffer = BytesIO()
        joblib.dump(self.best_model, buffer)
        return buffer.getvalue()
    
    def load_model(self, model_bytes):
        """Load a pre-trained model"""
        try:
            buffer = BytesIO(model_bytes)
            self.best_model = joblib.load(buffer)
            self.best_model_name = "Loaded Model"
            return True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def generate_model_report(self, results, problem_type):
        """Generate a comprehensive model performance report"""
        if not results:
            return None
        
        report = {
            'problem_type': problem_type,
            'best_model': self.best_model_name,
            'models_trained': len(results),
            'feature_count': self.X.shape[1],
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        # Add best model metrics
        if self.best_model_name in results:
            best_results = results[self.best_model_name]
            if problem_type == 'classification':
                report.update({
                    'test_accuracy': best_results['test_accuracy'],
                    'cv_accuracy': best_results['cv_mean'],
                    'cv_std': best_results['cv_std']
                })
            else:
                report.update({
                    'test_r2': best_results['test_r2'],
                    'test_mse': best_results['test_mse'],
                    'cv_r2': best_results['cv_mean'],
                    'cv_std': best_results['cv_std']
                })
        
        return report