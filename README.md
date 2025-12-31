# 🛡️ Fraudulent Job Posting Detection

A machine learning web application that detects fraudulent job postings using XGBoost.

## 🚀 Live Demo

**[Try the App →](https://fradulent-job-posting-detection.onrender.com)**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://fradulent-job-posting-detection.onrender.com)

---

## Project Structure

```
├── app.py                      # Flask web application
├── preprocessing_pipeline.pkl  # Trained preprocessing pipeline
├── best_model.pkl              # Trained XGBoost model
├── new.ipynb                   # Jupyter notebook with data analysis & model training
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   ├── index.html              # Main form page
│   ├── result.html             # Prediction result page
│   ├── 404.html                # Error page
│   └── 500.html                # Error page
├── categories/                 # Dropdown options for form fields
│   ├── industry.txt
│   ├── function.txt
│   ├── department.txt
│   └── ...
├── data/                       # Raw dataset
│   └── fake_job_postings 2.csv
└── vercel.json                 # Deployment config
```

## ✨ Features

- **Machine Learning Model**: XGBoost classifier trained on 17,880 job postings
- **Preprocessing Pipeline**: Automated feature engineering including:
  - Title feature extraction (11 features)
  - Target encoding for categorical variables
  - TF-IDF vectorization for text fields
  - Ordinal encoding for experience/education levels
- **Web Interface**: User-friendly form with searchable dropdowns
- **Real-time Prediction**: Instant fraud probability assessment

## 🛠️ Installation (Local Development)

1. Clone the repository:
```bash
git clone https://github.com/TusharM24/Fradulent-Job-Posting-Detection.git
cd Fradulent-Job-Posting-Detection
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open http://localhost:5000 in your browser

## 📊 Model Performance

| Metric | Description |
|--------|-------------|
| **Algorithm** | XGBoost (optimized for Recall) |
| **Primary Metric** | Recall (to minimize missed fraud cases) |
| **Training** | 5-fold Stratified Cross-Validation |
| **Class Balancing** | SMOTE applied to training data |

## 📝 Input Fields

| Field | Type | Description |
|-------|------|-------------|
| Job Title | Text | Required |
| Description | Text | Job description |
| Company Profile | Text | About the company |
| Requirements | Text | Job requirements |
| Benefits | Text | Job benefits |
| Country | Dropdown | 2-letter country code |
| State | Dropdown | State/region code |
| City | Text | City name |
| Industry | Dropdown | Industry sector |
| Function | Dropdown | Job function/domain |
| Department | Dropdown | Department name |
| Employment Type | Dropdown | Full-time, Part-time, etc. |
| Salary Range | Dropdown | Salary bracket |
| Required Experience | Dropdown | Experience level |
| Required Education | Dropdown | Education level |
| Telecommuting | Yes/No | Remote work available |
| Has Company Logo | Yes/No | Logo present in posting |
| Has Screening Questions | Yes/No | Application questions |

## 🔧 Technology Stack

- **Backend**: Flask (Python)
- **ML Libraries**: scikit-learn, XGBoost
- **Frontend**: HTML5, CSS3, JavaScript, jQuery, Select2
- **Data Processing**: pandas, numpy
- **Deployment**: Render

## 📄 License

MIT License
