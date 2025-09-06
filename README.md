#  VizualyticsAI


**Where Raw Data Becomes Intelligence**  

VizualyticsAI is an **end-to-end automated data analytics and machine learning platform** built with **Python** and **Streamlit**.  
It allows you to simply upload a dataset (CSV, Excel, JSON) and get:  

- ✅ Cleaned and processed data  
- 📊 Automated Exploratory Data Analysis (EDA)  
- 📈 Interactive visualizations  
- 🤖 Machine learning models (classification & regression)  
- 💬 Natural Language → SQL query generation  
- 📤 Easy exports (cleaned data, trained models, reports)  

---

## ✨ Features

- **📁 Data Upload**: Supports CSV, Excel, and JSON with preview & stats  
- **📊 Data Profiling**: Missing values, duplicates, data types, distributions  
- **🧹 Data Cleaning**: Null handling, duplicate removal, outlier detection  
- **📈 Auto Visualizations**: Histograms, correlations, categorical analysis  
- **🤖 AutoML Pipeline**: Detects problem type, trains multiple models, compares metrics  
- **💬 SQL Generator**: Convert natural language questions into SQL queries  
- **📤 Export System**: Download cleaned datasets & trained models  

---

## 🏗️ Tech Stack

- **Frontend / App**: [Streamlit](https://vizualyticsai.streamlit.app/)  
- **Data Processing**: `pandas`, `numpy`  
- **Visualization**: `matplotlib`, `seaborn`, `plotly`  
- **Machine Learning**: `scikit-learn`  
- **SQL Generation**: `sqlparse`, `transformers` (extensible)  
- **Deployment Ready**: Works locally or can be deployed to cloud (Heroku, AWS, etc.)  

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/DataForge-AI.git
cd DataForge-AI
```
### 2. Create Environment (Conda Recommended)
```bash
conda create -n dataforge-ai python=3.9
conda activate dataforge-ai
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
4. Run the Application
```bash
streamlit run app.py

```

📂 Project Structure
```bash
DataForge-AI/
├── app.py                # Main Streamlit app
├── config.py             # Branding & configuration
├── data_processor.py     # Data loading & cleaning
├── visualizer.py         # Visualization engine
├── ml_pipeline.py        # Machine learning pipeline
├── sql_generator.py      # Natural language → SQL
├── requirements.txt      # Dependencies
├── .gitignore            # Ignore unnecessary files
├── data/                 # Sample / uploaded data
├── exports/              # Exported cleaned files / reports
├── models/               # Saved ML models
└── logs/                 # Logs

```

## 📊 Example Workflow

1. Upload your dataset (`CSV`, `Excel`, or `JSON`)  
2. View profiling report (data types, missing values, duplicates)  
3. Clean your data with a single click (nulls, outliers, duplicates)  
4. Explore auto-generated **visualizations** (histograms, correlations, bar charts)  
5. Train ML models automatically (classification or regression detected)  
6. Ask questions in **natural language** → get SQL queries + results  
7. Export cleaned dataset & trained model  

---

## 🎨 Branding

- **App Name**: 🔨 DataForge AI  
- **Tagline**: *"Where Raw Data Becomes Intelligence"*  
- **Primary Colors**: Blue (#2E86AB) & Orange (#F24236)  

---

## 🛠️ Future Enhancements

- 🌐 Database integration (MySQL, PostgreSQL, MongoDB)  
- ⚡ Advanced AutoML (PyCaret, Auto-Sklearn, H2O)  
- 📡 API endpoints for programmatic access  
- ☁️ Cloud deployment (AWS/GCP/Azure)  
- 🔍 Real-time data pipelines  

---

## 🤝 Contributing

Contributions are welcome! Please fork this repo, create a new branch, and submit a pull request.  

---

## 📜 License

This project is licensed under the **MIT License** – feel free to use, modify, and distribute.  

