import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.data = None
        self.cleaned_data = None
        self.cleaning_log = []
        
    def load_data(self, uploaded_file):
        """Load data from uploaded file"""
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings and separators
                try:
                    self.data = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        self.data = pd.read_csv(uploaded_file, encoding='latin-1')
                    except:
                        self.data = pd.read_csv(uploaded_file, encoding='cp1252')
                
            elif file_extension in ['xlsx', 'xls']:
                self.data = pd.read_excel(uploaded_file)
                
            elif file_extension == 'json':
                self.data = pd.read_json(uploaded_file)
                
            else:
                st.error(f"Unsupported file format: {file_extension}")
                return False
            
            # Basic validation
            if self.data.empty:
                st.error("The uploaded file is empty!")
                return False
                
            # Clean column names
            self.data.columns = [col.strip().replace(' ', '_') for col in self.data.columns]
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def generate_profile_report(self):
        """Generate comprehensive data profile report"""
        if self.data is None:
            return None
        
        report = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns analysis
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_summary'] = self.data[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            report['categorical_summary'][col] = {
                'unique_values': self.data[col].nunique(),
                'top_values': self.data[col].value_counts().head(10).to_dict(),
                'missing_percentage': (self.data[col].isnull().sum() / len(self.data)) * 100
            }
        
        return report
    
    def clean_data(self, missing_strategy='drop', handle_duplicates=True, handle_outliers=False):
        """Comprehensive data cleaning pipeline"""
        if self.data is None:
            return ["Error: No data to clean"]
        
        if len(self.data) == 0:
            return ["Error: Dataset is empty"]
        
        self.cleaned_data = self.data.copy()
        self.cleaning_log = []
        
        original_shape = self.cleaned_data.shape
        
        try:
            # 1. Handle missing values
            self._handle_missing_values(missing_strategy)
            
            # Check if data still exists after missing value handling
            if len(self.cleaned_data) == 0:
                self.cleaning_log.append("Warning: All rows were removed due to missing values")
                return self.cleaning_log
            
            # 2. Handle duplicates
            if handle_duplicates:
                self._remove_duplicates()
            
            # Check if data still exists after duplicate removal
            if len(self.cleaned_data) == 0:
                self.cleaning_log.append("Warning: All rows were removed as duplicates")
                return self.cleaning_log
            
            # 3. Handle outliers
            if handle_outliers:
                self._handle_outliers()
            
            # Check if data still exists after outlier removal
            if len(self.cleaned_data) == 0:
                self.cleaning_log.append("Warning: All rows were removed as outliers")
                return self.cleaning_log
            
            # 4. Data type optimization (only if data exists)
            if len(self.cleaned_data) > 0:
                self._optimize_dtypes()
            
            # 5. Clean column names
            self._clean_column_names()
            
            final_shape = self.cleaned_data.shape
            self.cleaning_log.append(f"Overall: {original_shape[0]} → {final_shape[0]} rows ({original_shape[0] - final_shape[0]} removed)")
            
        except Exception as e:
            self.cleaning_log.append(f"Error during cleaning: {str(e)}")
            # Reset to original data if cleaning fails
            self.cleaned_data = self.data.copy()
        
        return self.cleaning_log
    
    def _handle_missing_values(self, strategy):
        """Handle missing values based on strategy"""
        missing_before = self.cleaned_data.isnull().sum().sum()
        
        if strategy == 'drop':
            self.cleaned_data = self.cleaned_data.dropna()
            self.cleaning_log.append(f"Dropped rows with missing values")
            
        elif strategy == 'fill_mean':
            numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
            self.cleaned_data[numeric_cols] = self.cleaned_data[numeric_cols].fillna(
                self.cleaned_data[numeric_cols].mean()
            )
            
            categorical_cols = self.cleaned_data.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                mode_value = self.cleaned_data[col].mode()
                if len(mode_value) > 0:
                    self.cleaned_data[col] = self.cleaned_data[col].fillna(mode_value[0])
                else:
                    self.cleaned_data[col] = self.cleaned_data[col].fillna('Unknown')
            
            self.cleaning_log.append(f"Filled missing values: numeric with mean, categorical with mode")
            
        elif strategy == 'fill_mode':
            for col in self.cleaned_data.columns:
                mode_value = self.cleaned_data[col].mode()
                if len(mode_value) > 0:
                    self.cleaned_data[col] = self.cleaned_data[col].fillna(mode_value[0])
            self.cleaning_log.append(f"Filled missing values with mode")
            
        elif strategy == 'fill_forward':
            self.cleaned_data = self.cleaned_data.fillna(method='ffill').fillna(method='bfill')
            self.cleaning_log.append(f"Forward filled missing values")
        
        missing_after = self.cleaned_data.isnull().sum().sum()
        if missing_before > 0:
            self.cleaning_log.append(f"Missing values: {missing_before} → {missing_after}")
    
    def _remove_duplicates(self):
        """Remove duplicate rows"""
        duplicates_before = self.cleaned_data.duplicated().sum()
        if duplicates_before > 0:
            self.cleaned_data = self.cleaned_data.drop_duplicates()
            self.cleaning_log.append(f"Removed {duplicates_before} duplicate rows")
    
    def _handle_outliers(self):
        """Remove outliers using IQR method"""
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            Q1 = self.cleaned_data[col].quantile(0.25)
            Q3 = self.cleaned_data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (self.cleaned_data[col] < lower_bound) | (self.cleaned_data[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                self.cleaned_data = self.cleaned_data[~outliers_mask]
                outliers_removed += outliers_count
                self.cleaning_log.append(f"Removed {outliers_count} outliers from {col}")
        
        if outliers_removed > 0:
            self.cleaning_log.append(f"Total outliers removed: {outliers_removed}")
    
    def _optimize_dtypes(self):
        """Optimize data types to reduce memory usage"""
        original_memory = self.cleaned_data.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        numeric_cols = self.cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_min = self.cleaned_data[col].min()
            col_max = self.cleaned_data[col].max()
            
            if self.cleaned_data[col].dtype == 'int64':
                if col_min >= -128 and col_max <= 127:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int8')
                elif col_min >= -32768 and col_max <= 32767:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int16')
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    self.cleaned_data[col] = self.cleaned_data[col].astype('int32')
            
            elif self.cleaned_data[col].dtype == 'float64':
                self.cleaned_data[col] = pd.to_numeric(self.cleaned_data[col], downcast='float')
        
        # Convert object columns with low cardinality to category
        object_cols = self.cleaned_data.select_dtypes(include=['object']).columns
        for col in object_cols:
            if self.cleaned_data[col].nunique() / len(self.cleaned_data) < 0.5:
                self.cleaned_data[col] = self.cleaned_data[col].astype('category')
        
        optimized_memory = self.cleaned_data.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100
        
        if memory_reduction > 5:
            self.cleaning_log.append(f"Memory optimized: {memory_reduction:.1f}% reduction")
    
    def _clean_column_names(self):
        """Clean and standardize column names"""
        new_columns = []
        for col in self.cleaned_data.columns:
            # Remove special characters and spaces
            clean_col = col.strip().lower()
            clean_col = ''.join(c if c.isalnum() else '_' for c in clean_col)
            # Remove multiple underscores
            clean_col = '_'.join(filter(None, clean_col.split('_')))
            new_columns.append(clean_col)
        
        if new_columns != list(self.cleaned_data.columns):
            self.cleaned_data.columns = new_columns
            self.cleaning_log.append("Column names standardized")
    
    def generate_data_quality_report(self):
        """Generate data quality metrics"""
        if self.cleaned_data is None:
            return None
        
        quality_metrics = {
            'completeness': (1 - self.cleaned_data.isnull().sum().sum() / 
                           (self.cleaned_data.shape[0] * self.cleaned_data.shape[1])) * 100,
            'uniqueness': (1 - self.cleaned_data.duplicated().sum() / len(self.cleaned_data)) * 100,
            'consistency': 100,  # Placeholder for consistency checks
            'validity': 100      # Placeholder for validity checks
        }
        
        # Display quality score
        overall_score = np.mean(list(quality_metrics.values()))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Completeness", f"{quality_metrics['completeness']:.1f}%")
        with col2:
            st.metric("Uniqueness", f"{quality_metrics['uniqueness']:.1f}%")
        with col3:
            st.metric("Consistency", f"{quality_metrics['consistency']:.1f}%")
        with col4:
            st.metric("Overall Score", f"{overall_score:.1f}%", 
                     delta="Good" if overall_score > 80 else "Needs Work")
        
        return quality_metrics
    
    def export_data(self, format_type="csv"):
        """Export cleaned data in various formats"""
        if self.cleaned_data is None:
            return None
        
        if format_type == "csv":
            return self.cleaned_data.to_csv(index=False).encode('utf-8')
        
        elif format_type == "excel":
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                self.cleaned_data.to_excel(writer, index=False, sheet_name='Cleaned_Data')
                
                # Add summary sheet
                summary_data = {
                    'Metric': ['Total Rows', 'Total Columns', 'Missing Values', 'Data Types'],
                    'Value': [
                        self.cleaned_data.shape[0],
                        self.cleaned_data.shape[1],
                        self.cleaned_data.isnull().sum().sum(),
                        len(self.cleaned_data.dtypes.unique())
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
            
            return output.getvalue()
        
        elif format_type == "json":
            return self.cleaned_data.to_json(orient='records', indent=2).encode('utf-8')
        
        elif format_type == "parquet":
            output = BytesIO()
            self.cleaned_data.to_parquet(output, index=False)
            return output.getvalue()
        
        return None
    
    def get_column_info(self):
        """Get detailed information about each column"""
        if self.cleaned_data is None:
            return None
        
        column_info = []
        for col in self.cleaned_data.columns:
            info = {
                'column': col,
                'dtype': str(self.cleaned_data[col].dtype),
                'non_null_count': self.cleaned_data[col].count(),
                'null_count': self.cleaned_data[col].isnull().sum(),
                'unique_count': self.cleaned_data[col].nunique(),
                'memory_usage': self.cleaned_data[col].memory_usage(deep=True)
            }
            
            if self.cleaned_data[col].dtype in ['int64', 'float64']:
                info.update({
                    'mean': self.cleaned_data[col].mean(),
                    'std': self.cleaned_data[col].std(),
                    'min': self.cleaned_data[col].min(),
                    'max': self.cleaned_data[col].max()
                })
            
            column_info.append(info)
        
        return pd.DataFrame(column_info)