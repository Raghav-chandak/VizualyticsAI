import streamlit as st
import pandas as pd
from datetime import datetime

# Simple imports - all files in root directory
from data_processor import DataProcessor
from visualizer import Visualizer
from ml_pipeline import MLPipeline
from sql_generator import SQLGenerator
import config

def main():
    # Page configuration
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #2E86AB 0%, #F24236 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title with custom styling
    st.markdown(f"""
    <div class="main-header">
        <h1>{config.APP_ICON} {config.APP_TITLE}</h1>
        <p>{config.APP_DESCRIPTION}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    if 'ml_pipeline' not in st.session_state:
        st.session_state.ml_pipeline = MLPipeline()
    if 'sql_generator' not in st.session_state:
        st.session_state.sql_generator = SQLGenerator()
    
    # Get instances
    processor = st.session_state.data_processor
    visualizer = st.session_state.visualizer
    ml_pipeline = st.session_state.ml_pipeline
    sql_gen = st.session_state.sql_generator
    
    # Sidebar navigation
    st.sidebar.header("🧭 Navigation")
    pages = {
        "📁 Data Upload": "upload",
        "📊 Data Profile": "profile",
        "🧹 Data Cleaning": "cleaning",
        "📈 Visualizations": "viz",
        "🤖 Machine Learning": "ml",
        "💾 SQL Generator": "sql",
        "📤 Export & Download": "export"
    }
    
    selected_page = st.sidebar.selectbox("Choose a section", list(pages.keys()))
    current_page = pages[selected_page]
    
    # Data status indicator
    if processor.data is not None:
        st.sidebar.success(f"✅ Data loaded: {processor.data.shape[0]} rows × {processor.data.shape[1]} cols")
        if processor.cleaned_data is not None:
            st.sidebar.info(f"🧹 Cleaned: {processor.cleaned_data.shape[0]} rows × {processor.cleaned_data.shape[1]} cols")
    else:
        st.sidebar.warning("⚠️ No data uploaded yet")
    
    # Page routing
    if current_page == "upload":
        render_upload_page(processor)
    elif current_page == "profile":
        render_profile_page(processor)
    elif current_page == "cleaning":
        render_cleaning_page(processor)
    elif current_page == "viz":
        render_visualization_page(processor, visualizer)
    elif current_page == "ml":
        render_ml_page(processor, ml_pipeline)
    elif current_page == "sql":
        render_sql_page(processor, sql_gen)
    elif current_page == "export":
        render_export_page(processor, ml_pipeline)

def render_upload_page(processor):
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=config.SUPPORTED_FORMATS,
            help=f"Supported formats: {', '.join(config.SUPPORTED_FORMATS)}"
        )
        
        if uploaded_file is not None:
            # File info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB",
                "File type": uploaded_file.type
            }
            
            st.json(file_details)
            
            if st.button("Load Data", type="primary"):
                with st.spinner("Loading data..."):
                    if processor.load_data(uploaded_file):
                        st.success("✅ Data loaded successfully!")
                        st.balloons()
                    else:
                        st.error("❌ Failed to load data")
    
    with col2:
        if processor.data is not None:
            st.subheader("📊 Data Preview")
            st.dataframe(processor.data.head(10))
            
            # Quick stats
            st.markdown(f"""
            <div class="metric-card">
                <h4>Quick Stats</h4>
                <p><strong>Shape:</strong> {processor.data.shape[0]} × {processor.data.shape[1]}</p>
                <p><strong>Memory:</strong> {processor.data.memory_usage(deep=True).sum() / 1024:.2f} KB</p>
                <p><strong>Columns:</strong> {len(processor.data.columns)}</p>
            </div>
            """, unsafe_allow_html=True)

def render_profile_page(processor):
    st.header("📊 Automated Data Profiling")
    
    if processor.data is not None:
        with st.spinner("Generating comprehensive profile..."):
            report = processor.generate_profile_report()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{report['shape'][0]:,}")
            with col2:
                st.metric("Columns", f"{report['shape'][1]:,}")
            with col3:
                st.metric("Duplicates", f"{report['duplicate_rows']:,}")
            with col4:
                missing_total = sum(report['missing_values'].values())
                st.metric("Missing Values", f"{missing_total:,}")
            
            # Detailed analysis
            tab1, tab2, tab3 = st.tabs(["🔍 Column Analysis", "📊 Statistical Summary", "🚨 Data Quality"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Data Types")
                    dtype_df = pd.DataFrame(list(report['dtypes'].items()), columns=['Column', 'Data Type'])
                    st.dataframe(dtype_df, use_container_width=True)
                
                with col2:
                    st.subheader("Missing Values by Column")
                    missing_df = pd.DataFrame(list(report['missing_values'].items()), 
                                            columns=['Column', 'Missing Count'])
                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                    if not missing_df.empty:
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("🎉 No missing values found!")
            
            with tab2:
                if report['numeric_summary']:
                    st.subheader("Numeric Columns Summary")
                    numeric_df = pd.DataFrame(report['numeric_summary'])
                    st.dataframe(numeric_df, use_container_width=True)
                
                if report['categorical_summary']:
                    st.subheader("Categorical Columns Summary")
                    for col, summary in report['categorical_summary'].items():
                        with st.expander(f"📋 {col}"):
                            st.write(f"**Unique Values:** {summary['unique_values']}")
                            st.write("**Top Values:**")
                            for value, count in summary['top_values'].items():
                                st.write(f"  • {value}: {count}")
            
            with tab3:
                processor.generate_data_quality_report()
    else:
        st.warning("⚠️ Please upload data first!")

def render_cleaning_page(processor):
    st.header("🧹 Automated Data Cleaning")
    
    if processor.data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_strategy = st.selectbox(
                "Missing Values Strategy",
                ["drop", "fill_mean", "fill_mode", "fill_forward"],
                help="Choose how to handle missing values"
            )
        
        with col2:
            handle_duplicates = st.checkbox("Remove Duplicates", value=True)
            handle_outliers = st.checkbox("Remove Outliers", value=False, 
                                        help="Uses IQR method")
        
        with col3:
            if st.button("🚀 Clean Data", type="primary"):
                with st.spinner("Cleaning data..."):
                    cleaning_log = processor.clean_data(
                        missing_strategy=missing_strategy,
                        handle_duplicates=handle_duplicates,
                        handle_outliers=handle_outliers
                    )
                    
                    if cleaning_log:
                        st.success("✅ Data cleaned successfully!")
                        
                        # Before/After comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Before Cleaning", f"{processor.data.shape[0]} rows")
                        with col2:
                            st.metric("After Cleaning", f"{processor.cleaned_data.shape[0]} rows")
                        
                        # Cleaning log
                        st.subheader("🔧 Cleaning Operations")
                        for log_entry in cleaning_log:
                            st.info(f"• {log_entry}")
                        
                        # Preview cleaned data
                        st.subheader("🔍 Cleaned Data Preview")
                        st.dataframe(processor.cleaned_data.head())
        
        # Data comparison
        if processor.cleaned_data is not None:
            st.subheader("📊 Before vs After Comparison")
            tab1, tab2 = st.tabs(["Original Data", "Cleaned Data"])
            
            with tab1:
                st.dataframe(processor.data.head())
                st.caption(f"Shape: {processor.data.shape}")
            
            with tab2:
                st.dataframe(processor.cleaned_data.head())
                st.caption(f"Shape: {processor.cleaned_data.shape}")
    else:
        st.warning("⚠️ Please upload data first!")

def render_visualization_page(processor, visualizer):
    st.header("📈 Automated Visualizations")
    
    if processor.cleaned_data is not None:
        with st.spinner("Generating visualizations..."):
            # Set data for visualizer
            visualizer.set_data(processor.cleaned_data)
            
            # Visualization options
            viz_types = st.multiselect(
                "Select Visualization Types",
                ["distributions", "correlations", "categorical", "time_series", "outliers"],
                default=["distributions", "correlations", "categorical"]
            )
            
            if viz_types:
                figures = visualizer.generate_visualizations(viz_types)
                
                for viz_info in figures:
                    st.subheader(f"📊 {viz_info['title']}")
                    st.plotly_chart(viz_info['figure'], use_container_width=True)
                    if 'description' in viz_info:
                        st.caption(viz_info['description'])
    else:
        st.warning("⚠️ Please clean data first!")

def render_ml_page(processor, ml_pipeline):
    st.header("🤖 Automated Machine Learning")
    
    if processor.cleaned_data is not None:
        # ML Configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            target_col = st.selectbox(
                "🎯 Select Target Column",
                options=list(processor.cleaned_data.columns)
            )
        
        with col2:
            problem_type = st.selectbox(
                "🔍 Problem Type",
                ["auto", "classification", "regression"],
                help="Auto will detect automatically"
            )
        
        with col3:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training multiple models..."):
                ml_pipeline.set_data(processor.cleaned_data)
                results = ml_pipeline.auto_ml_pipeline(
                    target_col=target_col,
                    problem_type=problem_type,
                    test_size=test_size
                )
                
                if results:
                    st.success(f"✅ Models trained! Problem: {results['problem_type']}")
                    
                    # Results display
                    st.subheader("📊 Model Performance")
                    
                    # Create performance comparison
                    performance_data = []
                    for model_name, metrics in results['results'].items():
                        if results['problem_type'] == 'classification':
                            performance_data.append({
                                'Model': model_name,
                                'Accuracy': f"{metrics['test_accuracy']:.4f}",
                                'Is Best': '🏆' if model_name == results['best_model'] else ''
                            })
                        else:
                            performance_data.append({
                                'Model': model_name,
                                'R² Score': f"{metrics['test_r2']:.4f}",
                                'MSE': f"{metrics['test_mse']:.4f}",
                                'Is Best': '🏆' if model_name == results['best_model'] else ''
                            })
                    
                    st.dataframe(pd.DataFrame(performance_data), use_container_width=True)
                    
                    # Feature importance if available
                    if 'feature_importance' in results:
                        st.subheader("🎯 Feature Importance")
                        importance_df = pd.DataFrame(results['feature_importance'])
                        st.bar_chart(importance_df.set_index('feature')['importance'])
    else:
        st.warning("⚠️ Please clean data first!")

def render_sql_page(processor, sql_gen):
    st.header("💾 Natural Language to SQL")
    
    if processor.cleaned_data is not None:
        sql_gen.set_data(processor.cleaned_data)
        
        # NL Query input
        query_text = st.text_area(
            "🗣️ Enter your question in natural language:",
            placeholder="e.g., What is the average salary by department?\nShow me the top 10 customers by revenue\nCount the number of orders per month",
            height=100
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("🔍 Generate SQL", type="primary"):
                if query_text:
                    sql_query = sql_gen.generate_sql_query(query_text)
                    st.session_state.generated_sql = sql_query
        
        # Display generated SQL
        if hasattr(st.session_state, 'generated_sql'):
            st.subheader("📝 Generated SQL Query")
            st.code(st.session_state.generated_sql, language='sql')
            
            # Execute query option
            if st.button("▶️ Execute Query"):
                try:
                    result = sql_gen.execute_query(st.session_state.generated_sql)
                    if result is not None:
                        st.subheader("📊 Query Results")
                        st.dataframe(result)
                except Exception as e:
                    st.error(f"Query execution failed: {e}")
        
        # Sample queries
        with st.expander("💡 Sample Queries"):
            st.markdown("""
            **Try these example queries:**
            - "Show me the first 10 rows"
            - "What is the average of all numeric columns?"
            - "Count rows by category"
            - "Find the maximum value in each column"
            - "Show unique values in the first column"
            """)
    else:
        st.warning("⚠️ Please clean data first!")

def render_export_page(processor, ml_pipeline):
    st.header("📤 Export & Download")
    
    if processor.cleaned_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Data Export")
            
            export_format = st.selectbox(
                "Export Format",
                ["CSV", "Excel", "JSON", "Parquet"]
            )
            
            if st.button("📥 Download Cleaned Data"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"cleaned_data_{timestamp}"
                
                if export_format == "CSV":
                    data_export = processor.export_data("csv")
                    st.download_button(
                        "⬇️ Download CSV",
                        data_export,
                        f"{filename}.csv",
                        "text/csv"
                    )
                elif export_format == "Excel":
                    data_export = processor.export_data("excel")
                    st.download_button(
                        "⬇️ Download Excel",
                        data_export,
                        f"{filename}.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.subheader("🤖 Model Export")
            
            if hasattr(ml_pipeline, 'best_model') and ml_pipeline.best_model is not None:
                st.success("✅ Trained model available")
                
                if st.button("📥 Download Model"):
                    model_data = ml_pipeline.export_model()
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        "⬇️ Download Model",
                        model_data,
                        f"model_{timestamp}.joblib",
                        "application/octet-stream"
                    )
            else:
                st.info("ℹ️ No trained model available. Train a model first!")
        
        # Export summary
        st.subheader("📋 Export Summary")
        summary_data = {
            "Original Data Shape": processor.data.shape if processor.data is not None else "N/A",
            "Cleaned Data Shape": processor.cleaned_data.shape,
            "Export Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "Data Quality Score": "Good" if processor.cleaned_data.isnull().sum().sum() == 0 else "Needs Review"
        }
        
        st.json(summary_data)
    else:
        st.warning("⚠️ Please clean data first!")

if __name__ == "__main__":
    main()