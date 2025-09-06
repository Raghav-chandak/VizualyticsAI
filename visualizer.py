import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class Visualizer:
    def __init__(self):
        self.data = None
        
    def set_data(self, data):
        """Set the data for visualization"""
        self.data = data
    
    def generate_visualizations(self, viz_types):
        """Generate multiple visualization types"""
        if self.data is None:
            return []
        
        figures = []
        
        if 'distributions' in viz_types:
            figures.extend(self._create_distribution_plots())
        
        if 'correlations' in viz_types:
            figures.extend(self._create_correlation_plots())
        
        if 'categorical' in viz_types:
            figures.extend(self._create_categorical_plots())
        
        if 'time_series' in viz_types:
            figures.extend(self._create_time_series_plots())
        
        if 'outliers' in viz_types:
            figures.extend(self._create_outlier_plots())
        
        return figures
    
    def _create_distribution_plots(self):
        """Create distribution plots for numeric columns"""
        figures = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Individual histograms
        for col in numeric_cols[:6]:  # Limit to first 6 columns
            fig = px.histogram(
                self.data, 
                x=col, 
                title=f'Distribution of {col.title()}',
                nbins=30,
                marginal='box',
                template='plotly_white'
            )
            fig.update_layout(
                showlegend=False,
                height=400
            )
            figures.append({
                'title': f'Distribution: {col.title()}',
                'figure': fig,
                'description': f'Histogram showing the distribution of {col} with box plot overlay'
            })
        
        # Combined distribution plot
        if len(numeric_cols) > 1:
            # Create subplots for multiple distributions
            cols_to_plot = min(4, len(numeric_cols))
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[col.title() for col in numeric_cols[:cols_to_plot]]
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            for i, col in enumerate(numeric_cols[:cols_to_plot]):
                row, col_pos = positions[i]
                fig.add_trace(
                    go.Histogram(x=self.data[col], name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title='Multiple Distributions Overview',
                height=600,
                template='plotly_white'
            )
            figures.append({
                'title': 'Multi-Column Distributions',
                'figure': fig,
                'description': 'Overview of distributions for multiple numeric columns'
            })
        
        return figures
    
    def _create_correlation_plots(self):
        """Create correlation analysis plots"""
        figures = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # Correlation heatmap
            corr_matrix = self.data[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap",
                template='plotly_white'
            )
            fig.update_layout(height=500)
            figures.append({
                'title': 'Correlation Heatmap',
                'figure': fig,
                'description': 'Correlation matrix showing relationships between numeric variables'
            })
            
            # Scatter plot matrix for highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if 0.5 <= corr_val < 1.0:  # Avoid perfect correlation (same variable)
                        high_corr_pairs.append((
                            corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_val
                        ))
            
            # Show top correlated pairs
            if high_corr_pairs:
                high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
                for i, (col1, col2, corr_val) in enumerate(high_corr_pairs[:3]):
                    fig = px.scatter(
                        self.data,
                        x=col1,
                        y=col2,
                        title=f'Scatter Plot: {col1.title()} vs {col2.title()} (r={corr_val:.3f})',
                        template='plotly_white',
                        trendline="ols"
                    )
                    fig.update_layout(height=400)
                    figures.append({
                        'title': f'Correlation: {col1} vs {col2}',
                        'figure': fig,
                        'description': f'Scatter plot showing correlation of {corr_val:.3f} between variables'
                    })
        
        return figures
    
    def _create_categorical_plots(self):
        """Create plots for categorical data"""
        figures = []
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
            if self.data[col].nunique() <= 20:  # Only if not too many unique values
                value_counts = self.data[col].value_counts().head(10)
                
                # Bar chart
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f'Distribution of {col.title()}',
                    labels={'x': col.title(), 'y': 'Count'},
                    template='plotly_white'
                )
                fig.update_layout(
                    showlegend=False,
                    height=400
                )
                figures.append({
                    'title': f'Category Distribution: {col.title()}',
                    'figure': fig,
                    'description': f'Bar chart showing frequency of categories in {col}'
                })
                
                # Pie chart for top categories
                if len(value_counts) > 1:
                    fig_pie = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f'Proportion of {col.title()}',
                        template='plotly_white'
                    )
                    fig_pie.update_layout(height=400)
                    figures.append({
                        'title': f'Category Proportions: {col.title()}',
                        'figure': fig_pie,
                        'description': f'Pie chart showing proportional distribution of {col}'
                    })
        
        # Cross-tabulation if multiple categorical columns
        if len(categorical_cols) >= 2:
            col1, col2 = categorical_cols[:2]
            if self.data[col1].nunique() <= 10 and self.data[col2].nunique() <= 10:
                crosstab = pd.crosstab(self.data[col1], self.data[col2])
                
                fig = px.imshow(
                    crosstab.values,
                    x=crosstab.columns,
                    y=crosstab.index,
                    text_auto=True,
                    title=f'Cross-tabulation: {col1.title()} vs {col2.title()}',
                    template='plotly_white',
                    aspect="auto"
                )
                fig.update_layout(height=400)
                figures.append({
                    'title': f'Cross-tabulation: {col1} vs {col2}',
                    'figure': fig,
                    'description': f'Heatmap showing relationship between {col1} and {col2}'
                })
        
        return figures
    
    def _create_time_series_plots(self):
        """Create time series plots if datetime columns exist"""
        figures = []
        
        # Look for datetime columns
        date_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    date_cols.append(col)
                except:
                    continue
        
        if date_cols:
            date_col = date_cols[0]
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for num_col in numeric_cols[:3]:  # First 3 numeric columns
                # Sort by date for proper time series
                data_sorted = self.data.sort_values(date_col)
                
                fig = px.line(
                    data_sorted,
                    x=date_col,
                    y=num_col,
                    title=f'Time Series: {num_col.title()} over Time',
                    template='plotly_white'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False
                )
                figures.append({
                    'title': f'Time Series: {num_col.title()}',
                    'figure': fig,
                    'description': f'Time series plot showing {num_col} trends over time'
                })
        
        return figures
    
    def _create_outlier_plots(self):
        """Create outlier detection plots"""
        figures = []
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Box plots for outlier detection
        if len(numeric_cols) > 0:
            # Individual box plots
            for col in numeric_cols[:4]:
                fig = px.box(
                    self.data,
                    y=col,
                    title=f'Outlier Detection: {col.title()}',
                    template='plotly_white'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False
                )
                figures.append({
                    'title': f'Outliers: {col.title()}',
                    'figure': fig,
                    'description': f'Box plot for outlier detection in {col}'
                })
            
            # Combined box plot
            if len(numeric_cols) > 1:
                # Normalize data for combined visualization
                normalized_data = self.data[numeric_cols].copy()
                for col in numeric_cols:
                    normalized_data[col] = (normalized_data[col] - normalized_data[col].mean()) / normalized_data[col].std()
                
                fig = px.box(
                    normalized_data[numeric_cols[:6]],  # Limit to 6 columns
                    title='Normalized Box Plots for Outlier Comparison',
                    template='plotly_white'
                )
                fig.update_layout(height=500)
                figures.append({
                    'title': 'Multi-Column Outlier Analysis',
                    'figure': fig,
                    'description': 'Normalized box plots for comparing outliers across columns'
                })
        
        return figures
    
    def create_custom_plot(self, plot_type, x_col=None, y_col=None, color_col=None):
        """Create custom plots based on user selection"""
        if self.data is None:
            return None
        
        try:
            if plot_type == 'scatter':
                fig = px.scatter(
                    self.data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f'Scatter Plot: {x_col} vs {y_col}',
                    template='plotly_white'
                )
            
            elif plot_type == 'line':
                fig = px.line(
                    self.data,
                    x=x_col,
                    y=y_col,
                    color=color_col,
                    title=f'Line Plot: {y_col} by {x_col}',
                    template='plotly_white'
                )
            
            elif plot_type == 'bar':
                if color_col:
                    data_grouped = self.data.groupby([x_col, color_col])[y_col].mean().reset_index()
                    fig = px.bar(
                        data_grouped,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f'Bar Plot: Average {y_col} by {x_col}',
                        template='plotly_white'
                    )
                else:
                    data_grouped = self.data.groupby(x_col)[y_col].mean().reset_index()
                    fig = px.bar(
                        data_grouped,
                        x=x_col,
                        y=y_col,
                        title=f'Bar Plot: Average {y_col} by {x_col}',
                        template='plotly_white'
                    )
            
            elif plot_type == 'histogram':
                fig = px.histogram(
                    self.data,
                    x=x_col,
                    color=color_col,
                    title=f'Histogram: {x_col}',
                    template='plotly_white'
                )
            
            else:
                return None
            
            fig.update_layout(height=500)
            return fig
            
        except Exception as e:
            st.error(f"Error creating plot: {e}")
            return None
    
    def generate_summary_dashboard(self):
        """Generate a comprehensive dashboard summary"""
        if self.data is None:
            return None
        
        # Create summary statistics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        
        summary_stats = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'missing_values': self.data.isnull().sum().sum(),
            'duplicate_rows': self.data.duplicated().sum(),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / (1024**2)
        }
        
        return summary_stats