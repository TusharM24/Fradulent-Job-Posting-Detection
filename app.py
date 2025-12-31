"""
Fraudulent Job Posting Detection - Flask Application
Uses the new preprocessing pipeline and best model from training.
"""

import pickle
import os
from flask import Flask, request, jsonify, render_template, Response, url_for
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================
# Custom Transformer Classes (required for pipeline loading)
# These must be defined BEFORE loading the pickled pipeline
# ============================================

class TitleFeatureTransformer(BaseEstimator, TransformerMixin):
    """Extracts features from title column."""
    
    def __init__(self, title_col='title'):
        self.title_col = title_col
        self.senior_keywords = ['senior', 'sr.', 'sr ', 'lead', 'head', 'director', 
                                'manager', 'chief', 'principal', 'vp', 'vice president']
        self.junior_keywords = ['junior', 'jr.', 'jr ', 'entry', 'intern', 'trainee', 
                                'graduate', 'assistant', 'associate', 'beginner']
        self.remote_keywords = ['remote', 'work from home', 'wfh', 'telecommute', 
                                'virtual', 'home-based', 'home based']
        self.urgent_keywords = ['urgent', 'immediate', 'asap', 'now hiring', 
                                'start immediately', 'hurry', 'limited time']
        self.parttime_keywords = ['part time', 'part-time', 'parttime', 'freelance', 
                                  'contract', 'temporary', 'temp ']
        self.suspicious_keywords = ['easy money', 'work from home', 'no experience', 
                                    'unlimited income', 'be your own boss', 'get paid daily']
    
    def fit(self, X, y=None):
        return self
    
    def _contains_any(self, text, keywords):
        text_lower = str(text).lower()
        return any(kw in text_lower for kw in keywords)
    
    def transform(self, X):
        X_copy = X.copy()
        titles = X_copy[self.title_col].fillna('').astype(str)
        
        X_copy['title_length'] = titles.str.len()
        X_copy['title_word_count'] = titles.str.split().str.len()
        X_copy['title_has_senior'] = titles.apply(lambda x: 1 if self._contains_any(x, self.senior_keywords) else 0)
        X_copy['title_has_junior'] = titles.apply(lambda x: 1 if self._contains_any(x, self.junior_keywords) else 0)
        X_copy['title_has_remote'] = titles.apply(lambda x: 1 if self._contains_any(x, self.remote_keywords) else 0)
        X_copy['title_has_urgent'] = titles.apply(lambda x: 1 if self._contains_any(x, self.urgent_keywords) else 0)
        X_copy['title_has_parttime'] = titles.apply(lambda x: 1 if self._contains_any(x, self.parttime_keywords) else 0)
        X_copy['title_has_suspicious'] = titles.apply(lambda x: 1 if self._contains_any(x, self.suspicious_keywords) else 0)
        X_copy['title_capital_ratio'] = titles.apply(lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))
        X_copy['title_special_char_ratio'] = titles.apply(lambda x: sum(1 for c in x if not c.isalnum() and c != ' ') / max(len(x), 1))
        X_copy['title_digit_count'] = titles.apply(lambda x: sum(1 for c in x if c.isdigit()))
        
        X_copy = X_copy.drop(columns=[self.title_col])
        return X_copy


class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    """Target encoding for categorical columns."""
    
    def __init__(self, columns, smoothing=10):
        self.columns = columns
        self.smoothing = smoothing
        self.encoding_maps = {}
        self.global_mean = None
    
    def fit(self, X, y):
        self.global_mean = y.mean()
        for col in self.columns:
            if col in X.columns:
                stats = pd.DataFrame({
                    'mean': y.groupby(X[col]).mean(),
                    'count': y.groupby(X[col]).count()
                })
                smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
                self.encoding_maps[col] = smoothed.to_dict()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(lambda x: self.encoding_maps[col].get(x, self.global_mean))
        return X_copy


class FrequencyEncoderTransformer(BaseEstimator, TransformerMixin):
    """Frequency encoding for high-cardinality categorical columns."""
    
    def __init__(self, columns, normalize=True):
        self.columns = columns
        self.normalize = normalize
        self.encoding_maps = {}
    
    def fit(self, X, y=None):
        for col in self.columns:
            if col in X.columns:
                counts = X[col].value_counts()
                if self.normalize:
                    self.encoding_maps[col] = (counts / len(X)).to_dict()
                else:
                    self.encoding_maps[col] = counts.to_dict()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].map(lambda x: self.encoding_maps[col].get(x, 0))
        return X_copy


class OrdinalEncoderTransformer(BaseEstimator, TransformerMixin):
    """Ordinal encoding for columns with natural order."""
    
    def __init__(self):
        self.experience_mapping = {
            'Unknown': 0, 'Not Applicable': 0,
            'Internship': 1, 'Entry level': 2, 'Associate': 3,
            'Mid-Senior level': 4, 'Director': 5, 'Executive': 6
        }
        self.education_mapping = {
            'Unknown': 0, 'Unspecified': 0,
            'Some High School Coursework': 1, 'High School or equivalent': 2,
            'Vocational': 3, 'Vocational - HS Diploma': 3,
            'Some College Coursework Completed': 4,
            'Vocational - Degree': 5, 'Certification': 5,
            'Associate Degree': 6, "Bachelor's Degree": 7,
            'Professional': 8, "Master's Degree": 9, 'Doctorate': 10
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if 'required_experience' in X_copy.columns:
            X_copy['required_experience'] = X_copy['required_experience'].map(
                lambda x: self.experience_mapping.get(x, 0)
            )
        if 'required_education' in X_copy.columns:
            X_copy['required_education'] = X_copy['required_education'].map(
                lambda x: self.education_mapping.get(x, 0)
            )
        return X_copy


class TextTfidfTransformer(BaseEstimator, TransformerMixin):
    """Combines text columns and applies TF-IDF vectorization."""
    
    def __init__(self, text_columns=['company_profile', 'description', 'requirements', 'benefits'],
                 max_features=50, ngram_range=(1, 2), min_df=5, max_df=0.90):
        self.text_columns = text_columns
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english'
        )
        self.feature_names = None
    
    def _combine_text(self, X):
        combined = X[self.text_columns[0]].fillna('').astype(str)
        for col in self.text_columns[1:]:
            if col in X.columns:
                combined = combined + ' ' + X[col].fillna('').astype(str)
        return combined.str.strip().str.replace(r'\s+', ' ', regex=True)
    
    def fit(self, X, y=None):
        combined_text = self._combine_text(X)
        self.vectorizer.fit(combined_text)
        self.feature_names = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        combined_text = self._combine_text(X_copy)
        
        tfidf_matrix = self.vectorizer.transform(combined_text)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=self.feature_names,
            index=X_copy.index
        )
        
        X_copy = X_copy.drop(columns=[c for c in self.text_columns if c in X_copy.columns])
        X_copy = X_copy.reset_index(drop=True)
        tfidf_df = tfidf_df.reset_index(drop=True)
        X_copy = pd.concat([X_copy, tfidf_df], axis=1)
        return X_copy


# ============================================
# Configuration
# ============================================
app = Flask(__name__)

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# Model and pipeline paths (check current directory first, then parent)
PIPELINE_PATH = os.path.join(BASE_DIR, 'preprocessing_pipeline.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')

# Fallback to parent directory (for development)
if not os.path.exists(PIPELINE_PATH):
    PIPELINE_PATH = os.path.join(PARENT_DIR, 'preprocessing_pipeline.pkl')
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PARENT_DIR, 'best_model.pkl')

print(f"Looking for pipeline at: {PIPELINE_PATH}")
print(f"Looking for model at: {MODEL_PATH}")

# ============================================
# Load Pipeline and Model
# ============================================
pipeline = None
model = None

try:
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    print(f"✓ Pipeline loaded from: {PIPELINE_PATH}")
except Exception as e:
    print(f"✗ Error loading pipeline: {e}")
    import traceback
    traceback.print_exc()

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from: {MODEL_PATH}")
    print(f"  Model type: {type(model).__name__}")
except Exception as e:
    print(f"✗ Error loading model: {e}")

# ============================================
# Dropdown Options (exact match with dataset)
# ============================================

# Required Experience - exact values from dataset
EXPERIENCE_OPTIONS = [
    'Unknown',
    'Internship',
    'Entry level',
    'Associate',
    'Mid-Senior level',
    'Director',
    'Executive',
    'Not Applicable'
]

# Required Education - exact values from dataset
EDUCATION_OPTIONS = [
    'Unknown',
    'Unspecified',
    'Some High School Coursework',
    'High School or equivalent',
    'Vocational',
    'Vocational - HS Diploma',
    'Vocational - Degree',
    'Some College Coursework Completed',
    'Certification',
    'Associate Degree',
    "Bachelor's Degree",
    'Professional',
    "Master's Degree",
    'Doctorate'
]

# Employment Type - exact values from dataset
EMPLOYMENT_TYPE_OPTIONS = [
    'Unknown',
    'Full-time',
    'Part-time',
    'Contract',
    'Temporary',
    'Other'
]

# Countries (most common, 2-letter codes)
COUNTRY_OPTIONS = [
    'Unknown', 'US', 'GB', 'GR', 'CA', 'DE', 'NZ', 'IN', 'AU', 'PH', 
    'NL', 'BE', 'IE', 'SG', 'HK', 'PL', 'EE', 'IL', 'FR', 'ES',
    'AE', 'EG', 'SE', 'RO', 'DK', 'ZA', 'BR', 'IT', 'FI', 'PK',
    'AT', 'CH', 'CZ', 'MY', 'NO', 'PT', 'RU', 'TH', 'TR', 'UA',
    'AR', 'CL', 'CO', 'MX', 'PE', 'VE', 'JP', 'KR', 'CN', 'ID'
]

# US States (2-letter codes)
US_STATE_OPTIONS = [
    'Unknown', 'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
]

# Expected column order (from model training)
# This must match the order used when the model was trained
EXPECTED_COLUMNS = [
    'department', 'salary_range', 'telecommuting', 'has_company_logo', 'has_questions',
    'employment_type', 'required_experience', 'required_education', 'industry', 'function',
    'country', 'state', 'city', 'title_length', 'title_word_count', 'title_has_senior',
    'title_has_junior', 'title_has_remote', 'title_has_urgent', 'title_has_parttime',
    'title_has_suspicious', 'title_capital_ratio', 'title_special_char_ratio', 'title_digit_count',
    'tfidf_ability', 'tfidf_amp', 'tfidf_based', 'tfidf_best', 'tfidf_business', 'tfidf_client',
    'tfidf_clients', 'tfidf_communication', 'tfidf_company', 'tfidf_customer', 'tfidf_customers',
    'tfidf_data', 'tfidf_design', 'tfidf_development', 'tfidf_environment', 'tfidf_experience',
    'tfidf_help', 'tfidf_high', 'tfidf_including', 'tfidf_job', 'tfidf_knowledge', 'tfidf_looking',
    'tfidf_management', 'tfidf_marketing', 'tfidf_new', 'tfidf_people', 'tfidf_position',
    'tfidf_product', 'tfidf_products', 'tfidf_provide', 'tfidf_quality', 'tfidf_required',
    'tfidf_sales', 'tfidf_service', 'tfidf_services', 'tfidf_skills', 'tfidf_software',
    'tfidf_solutions', 'tfidf_strong', 'tfidf_support', 'tfidf_team', 'tfidf_technical',
    'tfidf_technology', 'tfidf_time', 'tfidf_unknown', 'tfidf_web', 'tfidf_work',
    'tfidf_working', 'tfidf_world', 'tfidf_years'
]

# ============================================
# Helper Functions
# ============================================

def get_value_or_unknown(value, default='Unknown'):
    """Return 'Unknown' if value is empty, None, or blank."""
    if value is None or value == '' or (isinstance(value, str) and value.strip() == ''):
        return default
    return value.strip() if isinstance(value, str) else value


def prepare_input_dataframe(form_data):
    """
    Convert form data to DataFrame with exact column names matching the pipeline.
    Empty fields default to 'Unknown'.
    """
    # Extract and clean all fields
    data = {
        # Text fields - default to 'Unknown' if empty
        'title': get_value_or_unknown(form_data.get('job_title')),
        'company_profile': get_value_or_unknown(form_data.get('company-profile')),
        'description': get_value_or_unknown(form_data.get('job-description')),
        'requirements': get_value_or_unknown(form_data.get('job-requirements')),
        'benefits': get_value_or_unknown(form_data.get('benefits')),
        
        # Categorical fields (dropdowns) - already have proper values or 'Unknown'
        'industry': get_value_or_unknown(form_data.get('industry')),
        'function': get_value_or_unknown(form_data.get('domain')),
        'department': get_value_or_unknown(form_data.get('department')),
        'employment_type': get_value_or_unknown(form_data.get('employment-type')),
        'salary_range': get_value_or_unknown(form_data.get('salary-range')),
        'required_experience': get_value_or_unknown(form_data.get('required-experience')),
        'required_education': get_value_or_unknown(form_data.get('required-education')),
        
        # Location fields (split into country, state, city)
        'country': get_value_or_unknown(form_data.get('country')),
        'state': get_value_or_unknown(form_data.get('state')),
        'city': get_value_or_unknown(form_data.get('city')),
        
        # Binary fields (Yes/No -> 1/0)
        'telecommuting': 1 if form_data.get('telecommuting') == 'Yes' else 0,
        'has_company_logo': 1 if form_data.get('logo') == 'Yes' else 0,
        'has_questions': 1 if form_data.get('screening-questions') == 'Yes' else 0
    }
    
    # Create DataFrame with single row
    df = pd.DataFrame([data])
    
    return df


def predict_fraud(df):
    """
    Run prediction on preprocessed DataFrame.
    Returns prediction (0 or 1) and probability.
    """
    if pipeline is None or model is None:
        raise ValueError("Pipeline or model not loaded")
    
    # Preprocess
    X = pipeline.transform(df)
    
    # Reorder columns to match what model expects
    # This is critical because the model was trained with columns in a specific order
    X = X[EXPECTED_COLUMNS]
    
    print(f"Transformed shape: {X.shape}")
    print(f"Columns match expected: {list(X.columns) == EXPECTED_COLUMNS}")
    
    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else None
    
    return int(prediction), float(probability) if probability else 0.0


# ============================================
# API Routes for Dropdowns
# ============================================

@app.route('/api/categories/<category_name>', methods=['GET'])
def get_categories(category_name):
    """Serve category options for dropdowns."""
    category_map = {
        'required_experience': EXPERIENCE_OPTIONS,
        'required_education': EDUCATION_OPTIONS,
        'employment_type': EMPLOYMENT_TYPE_OPTIONS,
        'country': COUNTRY_OPTIONS,
        'state': US_STATE_OPTIONS,
        # For other categories, try loading from file
    }
    
    if category_name in category_map:
        return jsonify(category_map[category_name])
    
    # Try loading from categories folder
    category_file = os.path.join(BASE_DIR, 'categories', f'{category_name}.txt')
    if os.path.exists(category_file):
        try:
            with open(category_file, 'r', encoding='utf-8') as f:
                categories = [line.strip() for line in f if line.strip()]
            # Ensure 'Unknown' is first
            if 'Unknown' not in categories:
                categories.insert(0, 'Unknown')
            return jsonify(categories)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Category not found'}), 404


# ============================================
# Main Routes
# ============================================

@app.route('/')
def home():
    """Render the main form page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    # Get form data (support both GET and POST)
    form_data = request.form if request.method == 'POST' else request.args
    
    try:
        # Prepare input DataFrame
        input_df = prepare_input_dataframe(form_data)
        print("\n--- Input DataFrame ---")
        print(input_df.to_string())
        print(f"\nColumns: {list(input_df.columns)}")
        
        # Run prediction
        prediction, probability = predict_fraud(input_df)
        
        # Prepare result
        title = form_data.get('job_title', 'Unknown')
        department = form_data.get('department', 'Unknown')
        country = form_data.get('country', 'Unknown')
        state = form_data.get('state', 'Unknown')
        city = form_data.get('city', 'Unknown')
        location_display = f"{city}, {state}, {country}".replace('Unknown, ', '').replace(', Unknown', '')
        
        # Determine result
        is_fraud = prediction == 1
        result_text = 'FRAUDULENT' if is_fraud else 'LEGITIMATE'
        confidence = probability * 100 if is_fraud else (1 - probability) * 100
        
        # Render result page
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, {'#e74c3c' if is_fraud else '#27ae60'} 0%, {'#c0392b' if is_fraud else '#2ecc71'} 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 20px;
                }}
                .result-card {{
                    background: white;
                    border-radius: 20px;
                    padding: 40px;
                    max-width: 600px;
                    width: 100%;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    text-align: center;
                }}
                .icon {{
                    font-size: 80px;
                    margin-bottom: 20px;
                }}
                .result-title {{
                    font-size: 2em;
                    color: {'#e74c3c' if is_fraud else '#27ae60'};
                    margin-bottom: 10px;
                }}
                .confidence {{
                    font-size: 1.2em;
                    color: #666;
                    margin-bottom: 30px;
                }}
                .details {{
                    text-align: left;
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    margin-bottom: 30px;
                }}
                .details h3 {{
                    color: #333;
                    margin-bottom: 15px;
                }}
                .details p {{
                    margin: 8px 0;
                    color: #555;
                }}
                .details strong {{
                    color: #333;
                }}
                .back-btn {{
                    display: inline-block;
                    padding: 15px 40px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-weight: 600;
                    transition: transform 0.2s;
                }}
                .back-btn:hover {{
                    transform: translateY(-2px);
                }}
                .warning {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    padding: 15px;
                    border-radius: 10px;
                    margin-bottom: 20px;
                    color: #856404;
                }}
            </style>
        </head>
        <body>
            <div class="result-card">
                <div class="icon">{'🚨' if is_fraud else '✅'}</div>
                <h1 class="result-title">Job is {result_text}</h1>
                <p class="confidence">Confidence: {confidence:.1f}%</p>
                
                {'<div class="warning">⚠️ This job posting shows signs of being fraudulent. Please verify the employer before proceeding.</div>' if is_fraud else ''}
                
                <div class="details">
                    <h3>Job Details</h3>
                    <p><strong>Title:</strong> {title}</p>
                    <p><strong>Department:</strong> {department}</p>
                    <p><strong>Location:</strong> {location_display if location_display else 'Not specified'}</p>
                    <p><strong>Fraud Probability:</strong> {probability*100:.2f}%</p>
                </div>
                
                <a href="/" class="back-btn">← Analyze Another Job</a>
            </div>
        </body>
        </html>
        """
        
        return Response(html, mimetype='text/html')
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        
        error_html = f"""
        <!DOCTYPE html>
        <html>
        <head><title>Error</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 40px; background: #f8f9fa; }}
            .error-card {{ background: white; padding: 30px; border-radius: 10px; max-width: 600px; margin: 0 auto; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #e74c3c; }}
            pre {{ background: #f0f0f0; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            a {{ color: #667eea; }}
        </style>
        </head>
        <body>
            <div class="error-card">
                <h1>⚠️ Prediction Error</h1>
                <p>An error occurred while processing your request:</p>
                <pre>{str(e)}</pre>
                <p><a href="/">← Go Back</a></p>
            </div>
        </body>
        </html>
        """
        return Response(error_html, mimetype='text/html'), 500


# ============================================
# Run Application
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Fraudulent Job Posting Detection")
    print("="*50)
    print(f"Pipeline loaded: {'Yes' if pipeline else 'No'}")
    print(f"Model loaded: {'Yes' if model else 'No'}")
    print("="*50 + "\n")
    
    # Try port 5000, then 5001 if busy
    try:
        app.run(debug=True, port=5000)
    except OSError:
        print("Port 5000 busy, trying 5001...")
        app.run(debug=True, port=5001)
