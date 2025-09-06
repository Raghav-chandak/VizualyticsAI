import pandas as pd
import numpy as np
import sqlite3
import re
from io import StringIO
import streamlit as st

class SQLGenerator:
    def __init__(self):
        self.data = None
        self.table_name = "data_table"
        self.connection = None
        self.column_info = {}
        
    def set_data(self, data):
        """Set data and create in-memory database"""
        self.data = data
        self._create_database()
        self._analyze_columns()
    
    def _create_database(self):
        """Create SQLite in-memory database with the data"""
        try:
            self.connection = sqlite3.connect(':memory:')
            self.data.to_sql(self.table_name, self.connection, index=False, if_exists='replace')
        except Exception as e:
            st.error(f"Error creating database: {e}")
    
    def _analyze_columns(self):
        """Analyze columns for better query generation"""
        if self.data is None:
            return
        
        for col in self.data.columns:
            col_info = {
                'type': str(self.data[col].dtype),
                'unique_values': self.data[col].nunique(),
                'sample_values': list(self.data[col].dropna().unique()[:5]),
                'is_numeric': pd.api.types.is_numeric_dtype(self.data[col]),
                'is_datetime': pd.api.types.is_datetime64_any_dtype(self.data[col]),
                'has_nulls': self.data[col].isnull().any()
            }
            self.column_info[col] = col_info
    
    def generate_sql_query(self, natural_language_query):
        """Generate SQL query from natural language"""
        if self.data is None:
            return "-- No data available"
        
        query = natural_language_query.lower().strip()
        
        # Basic query patterns
        sql_query = self._match_query_patterns(query)
        
        if sql_query:
            return sql_query
        else:
            return self._generate_fallback_query(query)
    
    def _match_query_patterns(self, query):
        """Match common query patterns"""
        
        # Pattern 1: Count queries
        if any(keyword in query for keyword in ['count', 'how many', 'number of']):
            if 'group' in query or 'by' in query:
                # Group by query
                group_col = self._extract_column_name(query)
                if group_col:
                    return f"SELECT {group_col}, COUNT(*) as count FROM {self.table_name} GROUP BY {group_col} ORDER BY count DESC;"
            else:
                return f"SELECT COUNT(*) as total_rows FROM {self.table_name};"
        
        # Pattern 2: Average/Mean queries
        if any(keyword in query for keyword in ['average', 'avg', 'mean']):
            numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
            target_col = self._extract_column_name(query, numeric_cols)
            if target_col:
                return f"SELECT AVG({target_col}) as average_{target_col} FROM {self.table_name};"
            elif numeric_cols:
                avg_queries = [f"AVG({col}) as avg_{col}" for col in numeric_cols[:3]]
                return f"SELECT {', '.join(avg_queries)} FROM {self.table_name};"
        
        # Pattern 3: Maximum queries
        if any(keyword in query for keyword in ['max', 'maximum', 'highest', 'largest']):
            numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
            target_col = self._extract_column_name(query, numeric_cols)
            if target_col:
                return f"SELECT MAX({target_col}) as max_{target_col} FROM {self.table_name};"
            elif numeric_cols:
                return f"SELECT {', '.join([f'MAX({col}) as max_{col}' for col in numeric_cols[:3]])} FROM {self.table_name};"
        
        # Pattern 4: Minimum queries
        if any(keyword in query for keyword in ['min', 'minimum', 'lowest', 'smallest']):
            numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
            target_col = self._extract_column_name(query, numeric_cols)
            if target_col:
                return f"SELECT MIN({target_col}) as min_{target_col} FROM {self.table_name};"
            elif numeric_cols:
                return f"SELECT {', '.join([f'MIN({col}) as min_{col}' for col in numeric_cols[:3]])} FROM {self.table_name};"
        
        # Pattern 5: Sum queries
        if any(keyword in query for keyword in ['sum', 'total', 'add up']):
            numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
            target_col = self._extract_column_name(query, numeric_cols)
            if target_col:
                return f"SELECT SUM({target_col}) as total_{target_col} FROM {self.table_name};"
        
        # Pattern 6: Top/Bottom queries
        if any(keyword in query for keyword in ['top', 'first', 'highest']) and any(num in query for num in ['5', '10', '20']):
            limit = self._extract_number(query) or 10
            order_col = self._extract_column_name(query)
            if order_col and self.column_info.get(order_col, {}).get('is_numeric'):
                return f"SELECT * FROM {self.table_name} ORDER BY {order_col} DESC LIMIT {limit};"
            else:
                return f"SELECT * FROM {self.table_name} LIMIT {limit};"
        
        # Pattern 7: Unique values
        if any(keyword in query for keyword in ['unique', 'distinct', 'different']):
            target_col = self._extract_column_name(query)
            if target_col:
                return f"SELECT DISTINCT {target_col} FROM {self.table_name} ORDER BY {target_col};"
        
        # Pattern 8: Filter queries
        if any(keyword in query for keyword in ['where', 'filter', 'show me']) and '=' in query:
            condition = self._extract_condition(query)
            if condition:
                return f"SELECT * FROM {self.table_name} WHERE {condition};"
        
        # Pattern 9: Date-based queries
        if any(keyword in query for keyword in ['date', 'time', 'year', 'month', 'day']):
            date_cols = [col for col, info in self.column_info.items() if info['is_datetime']]
            if date_cols:
                date_col = date_cols[0]
                if 'group' in query:
                    return f"SELECT DATE({date_col}) as date, COUNT(*) as count FROM {self.table_name} GROUP BY DATE({date_col}) ORDER BY date;"
                else:
                    return f"SELECT * FROM {self.table_name} ORDER BY {date_col} DESC LIMIT 10;"
        
        return None
    
    def _extract_column_name(self, query, preferred_cols=None):
        """Extract column name from query"""
        columns = preferred_cols or list(self.data.columns)
        
        # Clean column names for matching
        query_words = re.findall(r'\b\w+\b', query.lower())
        
        # Direct column name match
        for col in columns:
            col_clean = col.lower().replace('_', ' ')
            if col_clean in query or col.lower() in query_words:
                return col
        
        # Partial match
        for col in columns:
            col_words = col.lower().replace('_', ' ').split()
            if any(word in query_words for word in col_words):
                return col
        
        # Return first preferred column if no match found
        return columns[0] if columns else None
    
    def _extract_number(self, query):
        """Extract number from query"""
        numbers = re.findall(r'\d+', query)
        return int(numbers[0]) if numbers else None
    
    def _extract_condition(self, query):
        """Extract WHERE condition from query"""
        # Simple pattern matching for conditions
        # This is basic and can be expanded
        
        # Look for column = value patterns
        for col in self.data.columns:
            if col.lower() in query:
                # Extract value after equals
                pattern = rf"{col.lower()}\s*=\s*['\"]?(\w+)['\"]?"
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    value = match.group(1)
                    if self.column_info[col]['is_numeric']:
                        return f"{col} = {value}"
                    else:
                        return f"{col} = '{value}'"
        
        return None
    
    def _generate_fallback_query(self, query):
        """Generate fallback query when no pattern matches"""
        # Default queries based on query content
        
        if any(keyword in query for keyword in ['show', 'display', 'see', 'view']):
            if 'all' in query:
                return f"SELECT * FROM {self.table_name};"
            else:
                return f"SELECT * FROM {self.table_name} LIMIT 10;"
        
        elif any(keyword in query for keyword in ['describe', 'summary', 'info']):
            numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
            if numeric_cols:
                stats = []
                for col in numeric_cols[:3]:
                    stats.extend([
                        f"AVG({col}) as avg_{col}",
                        f"MIN({col}) as min_{col}",
                        f"MAX({col}) as max_{col}"
                    ])
                return f"SELECT {', '.join(stats)} FROM {self.table_name};"
        
        # Default: show first 10 rows
        return f"SELECT * FROM {self.table_name} LIMIT 10;"
    
    def execute_query(self, sql_query):
        """Execute SQL query and return results"""
        if self.connection is None:
            st.error("No database connection available!")
            return None
        
        try:
            # Clean the query
            sql_query = sql_query.strip()
            if not sql_query.endswith(';'):
                sql_query += ';'
            
            # Execute query
            result = pd.read_sql_query(sql_query, self.connection)
            return result
            
        except Exception as e:
            st.error(f"Query execution error: {e}")
            return None
    
    def get_table_schema(self):
        """Get table schema information"""
        if self.connection is None:
            return None
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name})")
            schema = cursor.fetchall()
            
            schema_df = pd.DataFrame(schema, columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk'])
            return schema_df
            
        except Exception as e:
            st.error(f"Error getting schema: {e}")
            return None
    
    def suggest_queries(self):
        """Suggest common useful queries"""
        if self.data is None:
            return []
        
        suggestions = []
        
        # Basic queries
        suggestions.append({
            'description': 'Show first 10 rows',
            'query': f"SELECT * FROM {self.table_name} LIMIT 10;",
            'category': 'Basic'
        })
        
        suggestions.append({
            'description': 'Count total rows',
            'query': f"SELECT COUNT(*) as total_rows FROM {self.table_name};",
            'category': 'Basic'
        })
        
        # Numeric column suggestions
        numeric_cols = [col for col, info in self.column_info.items() if info['is_numeric']]
        for col in numeric_cols[:3]:
            suggestions.append({
                'description': f'Average of {col}',
                'query': f"SELECT AVG({col}) as avg_{col} FROM {self.table_name};",
                'category': 'Statistics'
            })
            
            suggestions.append({
                'description': f'Top 5 highest {col}',
                'query': f"SELECT * FROM {self.table_name} ORDER BY {col} DESC LIMIT 5;",
                'category': 'Analysis'
            })
        
        # Categorical column suggestions
        categorical_cols = [col for col, info in self.column_info.items() if not info['is_numeric'] and info['unique_values'] <= 20]
        for col in categorical_cols[:2]:
            suggestions.append({
                'description': f'Count by {col}',
                'query': f"SELECT {col}, COUNT(*) as count FROM {self.table_name} GROUP BY {col} ORDER BY count DESC;",
                'category': 'Grouping'
            })
            
            suggestions.append({
                'description': f'Unique values in {col}',
                'query': f"SELECT DISTINCT {col} FROM {self.table_name} ORDER BY {col};",
                'category': 'Exploration'
            })
        
        # Missing data queries
        nullable_cols = [col for col, info in self.column_info.items() if info['has_nulls']]
        if nullable_cols:
            suggestions.append({
                'description': 'Rows with missing data',
                'query': f"SELECT * FROM {self.table_name} WHERE {' IS NULL OR '.join(nullable_cols[:3])} IS NULL;",
                'category': 'Data Quality'
            })
        
        # Date-based queries
        date_cols = [col for col, info in self.column_info.items() if info['is_datetime']]
        if date_cols:
            date_col = date_cols[0]
            suggestions.append({
                'description': f'Data by date ({date_col})',
                'query': f"SELECT DATE({date_col}) as date, COUNT(*) as records FROM {self.table_name} GROUP BY DATE({date_col}) ORDER BY date DESC;",
                'category': 'Time Series'
            })
        
        return suggestions
    
    def validate_query(self, sql_query):
        """Validate SQL query for safety"""
        dangerous_keywords = [
            'DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 
            'TRUNCATE', 'EXEC', 'EXECUTE', 'UNION'
        ]
        
        query_upper = sql_query.upper()
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Query contains potentially dangerous keyword: {keyword}"
        
        return True, "Query is safe"
    
    def explain_query(self, sql_query):
        """Provide explanation of what the SQL query does"""
        query_lower = sql_query.lower().strip()
        
        explanations = []
        
        # Basic operation
        if query_lower.startswith('select'):
            if 'count(*)' in query_lower:
                explanations.append("Counts the number of rows")
            elif 'avg(' in query_lower:
                explanations.append("Calculates average values")
            elif 'max(' in query_lower:
                explanations.append("Finds maximum values")
            elif 'min(' in query_lower:
                explanations.append("Finds minimum values")
            elif 'sum(' in query_lower:
                explanations.append("Calculates sum of values")
            elif 'distinct' in query_lower:
                explanations.append("Shows unique values")
            else:
                explanations.append("Retrieves data from the table")
        
        # Filtering
        if 'where' in query_lower:
            explanations.append("Filters data based on conditions")
        
        # Grouping
        if 'group by' in query_lower:
            explanations.append("Groups data by specified columns")
        
        # Ordering
        if 'order by' in query_lower:
            if 'desc' in query_lower:
                explanations.append("Orders results in descending order")
            else:
                explanations.append("Orders results in ascending order")
        
        # Limiting
        if 'limit' in query_lower:
            limit_match = re.search(r'limit\s+(\d+)', query_lower)
            if limit_match:
                limit_num = limit_match.group(1)
                explanations.append(f"Limits results to {limit_num} rows")
        
        return ". ".join(explanations) + "." if explanations else "Executes a custom SQL query."
    
    def get_query_complexity(self, sql_query):
        """Analyze query complexity"""
        query_lower = sql_query.lower()
        complexity_score = 0
        complexity_factors = []
        
        # Basic operations
        if 'select' in query_lower:
            complexity_score += 1
        
        # Joins (not implemented in basic version but good to have)
        if any(join_type in query_lower for join_type in ['join', 'inner join', 'left join', 'right join']):
            complexity_score += 3
            complexity_factors.append("Uses table joins")
        
        # Subqueries
        if query_lower.count('select') > 1:
            complexity_score += 4
            complexity_factors.append("Contains subqueries")
        
        # Aggregations
        agg_functions = ['count', 'sum', 'avg', 'min', 'max', 'group by']
        agg_count = sum(1 for func in agg_functions if func in query_lower)
        if agg_count > 0:
            complexity_score += agg_count * 2
            complexity_factors.append(f"Uses {agg_count} aggregation function(s)")
        
        # Conditions
        if 'where' in query_lower:
            complexity_score += 2
            complexity_factors.append("Has filtering conditions")
        
        # Determine complexity level
        if complexity_score <= 3:
            level = "Simple"
        elif complexity_score <= 8:
            level = "Moderate"
        else:
            level = "Complex"
        
        return {
            'level': level,
            'score': complexity_score,
            'factors': complexity_factors
        }
    
    def optimize_query(self, sql_query):
        """Suggest query optimizations"""
        suggestions = []
        query_lower = sql_query.lower()
        
        # Suggest using LIMIT for large datasets
        if 'limit' not in query_lower and 'count' not in query_lower:
            suggestions.append("Consider adding LIMIT to prevent large result sets")
        
        # Suggest specific columns instead of *
        if 'select *' in query_lower:
            suggestions.append("Consider selecting specific columns instead of * for better performance")
        
        # Suggest indexing for WHERE clauses
        if 'where' in query_lower:
            suggestions.append("For large datasets, ensure WHERE clause columns are indexed")
        
        # Suggest HAVING vs WHERE for aggregated data
        if 'group by' in query_lower and 'where' in query_lower:
            # Check if WHERE is used on aggregated columns
            if any(func in query_lower for func in ['count(', 'sum(', 'avg(', 'min(', 'max(']):
                suggestions.append("Use HAVING instead of WHERE for filtering aggregated results")
        
        return suggestions
    
    def generate_sample_data_queries(self):
        """Generate queries to explore the data structure"""
        queries = []
        
        if self.data is None:
            return queries
        
        # Data overview
        queries.append({
            'title': 'Data Overview',
            'query': f"SELECT COUNT(*) as total_rows, COUNT(DISTINCT *) as unique_rows FROM {self.table_name};",
            'description': 'Get basic data statistics'
        })
        
        # Column information
        for col in list(self.data.columns)[:5]:  # First 5 columns
            col_info = self.column_info[col]
            
            if col_info['is_numeric']:
                queries.append({
                    'title': f'{col} Statistics',
                    'query': f"SELECT MIN({col}) as min_val, MAX({col}) as max_val, AVG({col}) as avg_val, COUNT({col}) as non_null_count FROM {self.table_name};",
                    'description': f'Statistical summary of {col}'
                })
            else:
                queries.append({
                    'title': f'{col} Distribution',
                    'query': f"SELECT {col}, COUNT(*) as frequency FROM {self.table_name} GROUP BY {col} ORDER BY frequency DESC LIMIT 10;",
                    'description': f'Top 10 most frequent values in {col}'
                })
        
        return queries
    
    def close_connection(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None