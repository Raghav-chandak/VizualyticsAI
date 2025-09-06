import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EXPORTS_DIR = BASE_DIR / "exports"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, EXPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# App configuration
APP_TITLE = "AI-Powered Data Analytics Platform"
APP_ICON = "🚀"
APP_DESCRIPTION = "Upload your data and get comprehensive insights, automated cleaning, visualizations, and ML predictions!"

# File handling
MAX_FILE_SIZE_MB = 200
SUPPORTED_FORMATS = ['csv', 'xlsx', 'xls', 'json', 'parquet']
ENCODING_OPTIONS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

# Data processing configuration
MISSING_VALUE_STRATEGIES = {
    'drop': 'Drop rows with missing values',
    'fill_mean': 'Fill numeric with mean, categorical with mode',
    'fill_mode': 'Fill all columns with mode',
    'fill_forward': 'Forward fill missing values',
    'fill_zero': 'Fill missing values with zero',
    'interpolate': 'Interpolate missing values'
}

# ML configuration
RANDOM_STATE = 42
TEST_SIZE_OPTIONS = [0.1, 0.15, 0.2, 0.25, 0.3]
DEFAULT_TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_CATEGORIES = 20  # Maximum unique values for categorical encoding

# Model configuration
CLASSIFICATION_MODELS = {
    'Random Forest': {
        'name': 'RandomForestClassifier',
        'params': {'n_estimators': 100, 'random_state': RANDOM_STATE}
    },
    'Logistic Regression': {
        'name': 'LogisticRegression',
        'params': {'random_state': RANDOM_STATE, 'max_iter': 1000}
    },
    'Gradient Boosting': {
        'name': 'GradientBoostingClassifier',
        'params': {'random_state': RANDOM_STATE}
    },
    'K-Nearest Neighbors': {
        'name': 'KNeighborsClassifier',
        'params': {'n_neighbors': 5}
    },
    'Support Vector Machine': {
        'name': 'SVC',
        'params': {'random_state': RANDOM_STATE, 'probability': True}
    }
}

REGRESSION_MODELS = {
    'Random Forest': {
        'name': 'RandomForestRegressor',
        'params': {'n_estimators': 100, 'random_state': RANDOM_STATE}
    },
    'Linear Regression': {
        'name': 'LinearRegression',
        'params': {}
    },
    'Ridge Regression': {
        'name': 'Ridge',
        'params': {'random_state': RANDOM_STATE}
    },
    'Lasso Regression': {
        'name': 'Lasso',
        'params': {'random_state': RANDOM_STATE}
    },
    'Gradient Boosting': {
        'name': 'GradientBoostingRegressor',
        'params': {'random_state': RANDOM_STATE}
    },
    'K-Nearest Neighbors': {
        'name': 'KNeighborsRegressor',
        'params': {'n_neighbors': 5}
    },
    'Support Vector Machine': {
        'name': 'SVR',
        'params': {}
    }
}

# Visualization configuration
PLOT_TEMPLATE = 'plotly_white'
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
CHART_HEIGHT = 400
CHART_WIDTH = None  # Auto-width

VIZ_TYPES = {
    'distributions': 'Distribution Plots',
    'correlations': 'Correlation Analysis',
    'categorical': 'Categorical Analysis',
    'time_series': 'Time Series Analysis',
    'outliers': 'Outlier Detection',
    'custom': 'Custom Plots'
}

# Data quality thresholds
QUALITY_THRESHOLDS = {
    'completeness': 0.8,  # 80% non-missing data
    'uniqueness': 0.95,   # 95% unique rows
    'consistency': 0.9,   # 90% consistent formats
    'validity': 0.85      # 85% valid values
}

# SQL configuration
SQL_TABLE_NAME = "data_table"
MAX_QUERY_RESULTS = 1000
SAFE_SQL_KEYWORDS = [
    'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
    'LIMIT', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX',
    'AS', 'AND', 'OR', 'NOT', 'IN', 'LIKE', 'BETWEEN'
]

DANGEROUS_SQL_KEYWORDS = [
    'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER',
    'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION', 'GRANT', 'REVOKE'
]

# Export configuration
EXPORT_FORMATS = {
    'csv': {
        'extension': '.csv',
        'mime_type': 'text/csv',
        'description': 'Comma Separated Values'
    },
    'excel': {
        'extension': '.xlsx',
        'mime_type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'description': 'Microsoft Excel'
    },
    'json': {
        'extension': '.json',
        'mime_type': 'application/json',
        'description': 'JavaScript Object Notation'
    },
    'parquet': {
        'extension': '.parquet',
        'mime_type': 'application/octet-stream',
        'description': 'Apache Parquet'
    }
}

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
MAX_LOG_SIZE_MB = 10

# UI configuration
SIDEBAR_WIDTH = 300
MAIN_CONTENT_PADDING = "1rem"

# Theme colors
THEME_COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'success': '#28a745',
    'warning': '#ffc107',
    'error': '#dc3545',
    'info': '#17a2b8'
}

# Feature flags
ENABLE_ADVANCED_ML = True
ENABLE_AUTO_EDA = True
ENABLE_SQL_GENERATOR = True
ENABLE_CUSTOM_PLOTS = True
ENABLE_MODEL_EXPORT = True
ENABLE_DATA_PROFILING = True

# Performance settings
MAX_ROWS_FOR_CORRELATION = 10000  # Skip correlation for datasets larger than this
MAX_CATEGORIES_FOR_PLOTS = 50     # Skip categorical plots for high cardinality
SAMPLE_SIZE_FOR_LARGE_DATA = 10000  # Sample size for large datasets

# Memory management
MEMORY_THRESHOLD_MB = 500  # Warning threshold for memory usage
AUTO_GARBAGE_COLLECTION = True

# Advanced features configuration
OUTLIER_DETECTION_METHODS = ['IQR', 'Z-Score', 'Isolation Forest']
DEFAULT_OUTLIER_METHOD = 'IQR'

IMPUTATION_STRATEGIES = {
    'numeric': ['mean', 'median', 'mode', 'constant', 'knn'],
    'categorical': ['mode', 'constant', 'new_category']
}

# API settings (for future API integration)
API_VERSION = "v1"
API_RATE_LIMIT = 100  # requests per minute
API_TIMEOUT = 30  # seconds

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds
ENABLE_CACHING = True

# Development settings
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
PROFILING_ENABLED = os.getenv('PROFILING', 'False').lower() == 'true'

# Error messages
ERROR_MESSAGES = {
    'file_too_large': f'File size exceeds {MAX_FILE_SIZE_MB}MB limit',
    'unsupported_format': f'Unsupported file format. Supported: {", ".join(SUPPORTED_FORMATS)}',
    'no_data': 'No data available. Please upload a file first.',
    'invalid_target': 'Invalid target column selected.',
    'model_training_failed': 'Model training failed. Please check your data.',
    'query_execution_failed': 'SQL query execution failed.',
    'export_failed': 'Data export failed.'
}

# Success messages
SUCCESS_MESSAGES = {
    'data_loaded': 'Data loaded successfully!',
    'data_cleaned': 'Data cleaning completed!',
    'model_trained': 'Models trained successfully!',
    'query_executed': 'Query executed successfully!',
    'data_exported': 'Data exported successfully!'
}

# Help text
HELP_TEXT = {
    'file_upload': 'Upload CSV, Excel, JSON, or Parquet files. Maximum size: 200MB.',
    'missing_values': 'Choose how to handle missing values in your dataset.',
    'target_column': 'Select the column you want to predict (target variable).',
    'problem_type': 'Auto-detect will determine if it\'s classification or regression.',
    'test_size': 'Percentage of data to use for testing the model.',
    'sql_query': 'Enter natural language questions like "What is the average salary?" or "Show me the top 10 customers".'
}

# Version info
VERSION = "1.0.0"
LAST_UPDATED = "2024-12-19"
AUTHOR = "DataForge AI Team"
TAGLINE = "Where Raw Data Becomes Intelligence"