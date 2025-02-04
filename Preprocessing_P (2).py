from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.tree import DecisionTreeClassifier
import json
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeRegressor

high_cardinality_cols = ['title', 'location', 'department', 'industry','salary_range','function']
descriptive_columns = ['company_profile', 'description', 'requirements', 'benefits']
subset_features=['company_profile_encoded',
 'has_questions',
 'benefits_encoded',
 'location_US, TX, Houston',
 'required_experience_Mid-Senior level',
 "required_education_Bachelor's Degree",
 'has_company_logo',
 'department_Other',
 'required_experience_Entry level',
 'salary_range_Other',
 'function_Other',
 'industry_Other',
 'department_Unknown',
 'function_Unknown',
 'function_Engineering',
 'salary_range_Unknown',
 'required_experience_Unknown',
 'requirements_encoded',
 'industry_Hospital & Health Care',
 'required_education_High School or equivalent',
 'location_Other',
 'employment_type_Unknown',
 'industry_Unknown',
 'required_education_Some High School Coursework',
 'employment_type_Full-time',
 'industry_Marketing and Advertising',
 'employment_type_Part-time',
 'function_Customer Service',
 'function_Sales',
 'required_education_Unknown',
 'telecommuting',
 'required_education_Unspecified',
 'industry_Information Technology and Services',
 'location_US, , ',
 'industry_Financial Services',
 'required_experience_Director',
 'clients',
 'required_experience_Not Applicable',
 'position',
 'required_experience_Internship',
 'location_US, CA, San Francisco',
 'industry_Consumer Services',
 "required_education_Master's Degree",
 'department_Sales',
 'process',
 'required',
 'department_Information Technology',
 'required_education_Certification',
 'amp',
 'location_Unknown',
 'location_US, NY, New York',
 'office',
 'employment_type_Other',
 'function_Information Technology',
 'required_experience_Executive',
 'salary_range_30000-40000',
 'title_Other',
 'department_Engineering',
 'function_Health Care Provider',
 'required_education_Some College Coursework Completed']

common_words={'new', 'including', 'sales', '-', '&amp;', 'development', 'company', 'project', 'job', 'people', 'support', 'solutions', 'design', 'work', 'customer', 'looking', 'service', 'data', 'services', 'knowledge', 'high', 'business', 'product', 'skills', 'technology', 'unknown', 'team', 'help', 'working', 'ability', 'quality', 'time', 'technical', 'management', 'provide', 'experience', 'years', 'strong', 'communication'}

# Filling missing values 
class FillMissingValues(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self    
    def transform(self, X):
        return X.fillna('Unknown')

# Creating missing indicators for text fields
class CreateMissingIndicators(BaseEstimator, TransformerMixin):
    def __init__(self, descriptive_columns):
        self.descriptive_columns = descriptive_columns       
    def fit(self, X, y=None):
        return self   
    def transform(self, X):
        for col in self.descriptive_columns:
            X[f'{col}_encoded'] = X[col].apply(lambda x: 1 if x != 'Unknown' else 0)
        return X

# Using TF-IDF for vectorizing the text fields
class VectorizeDescriptiveText(BaseEstimator, TransformerMixin):
    def __init__(self, descriptive_columns, max_features=20, common_words=None):
        self.descriptive_columns = descriptive_columns
        self.max_features = max_features
        self.common_words = common_words
        self.vectorizer = None  

# combining all columns and finding the list of stop words
    def fit(self, X, y=None):
        X['combined_text'] = X[self.descriptive_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        exclude_words = list(ENGLISH_STOP_WORDS.union(self.common_words)) if self.common_words else list(ENGLISH_STOP_WORDS)
        exclude_words.append('industry')
        self.vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words=exclude_words)
        self.vectorizer.fit(X['combined_text'])
        
        return self

    def transform(self, X):
        if self.vectorizer is None:
            raise ValueError("fit has not been called before transform")
        
        X['combined_text'] = X[self.descriptive_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        text_vectors = self.vectorizer.transform(X['combined_text'])
        text_vectors_dense = text_vectors.toarray()
        feature_names = self.vectorizer.get_feature_names_out()
        tfidf_df = pd.DataFrame(text_vectors_dense, columns=feature_names)
        X = X.drop(columns=self.descriptive_columns + ['combined_text'])
        X_reset = X.reset_index(drop=True)
        tfidf_df_reset = tfidf_df.reset_index(drop=True)
        Ndf = pd.concat([X_reset, tfidf_df_reset], axis=1)
        
        return Ndf

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class CollapseHighCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, columns, top_n=8):
        self.columns = columns
        self.top_n = top_n
    
    def fit(self, X, y=None):
        self.top_values = {}

        for column in self.columns:
            self.top_values[column] = list(X[column].value_counts().nlargest(self.top_n).index)

        return self
    
    def transform(self, X):
        for column in self.columns:
            X[column] = X[column].apply(lambda x: x if x in self.top_values[column] else 'Other')
        return X
    
class OneHotEncodeLowCardinality(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
        self.encoder = OneHotEncoder(sparse_output=False, drop='first') 

# Identify lowing cardinality columns    
    def fit(self, X, y=None):
        self.low_cardinality_cols = [col for col in X.columns if X[col].nunique() > 2 and X[col].dtype == 'object']
        self.encoder.fit(X[self.low_cardinality_cols])
        return self

# Changing the encoded array to dataframe and combining it with the original dataframe    
    def transform(self, X):
        encoded = self.encoder.transform(X[self.low_cardinality_cols])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.low_cardinality_cols))
        X = X.drop(columns=self.low_cardinality_cols)
        X = pd.concat([X, encoded_df], axis=1)
        
        return X

class ConvertToIntegers(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

# Performing feature selection
class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.selected_features]

#Combining all transformers into one pipleline
preprocessing_pipeline = Pipeline([
    ('fill_missing_values', FillMissingValues()),
    ('create_missing_indicators', CreateMissingIndicators(descriptive_columns)), 
    ('vectorize_text', VectorizeDescriptiveText(descriptive_columns, max_features=20, common_words=common_words)),  
    ('collapse_high_cardinality', CollapseHighCardinality(columns=high_cardinality_cols , top_n=9)),
    ('one_hot_encode', OneHotEncodeLowCardinality(max_categories=20)), 
    ('convert_to_integers', ConvertToIntegers()), 
    ('feature_selection', FeatureSelection(subset_features)) 
])

class RandomForest:
    def __init__(self, n_estimators=200, max_depth=20, random_state=50):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

# Training the model by fitting n_estimators 
    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        self.feature_importances_ = np.zeros(X.shape[1]) 
        
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X.iloc[indices]  
            y_bootstrap = y.iloc[indices]              
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= self.n_estimators

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_votes = [np.bincount(pred).argmax() for pred in predictions.T]
        return np.array(majority_votes)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

# Training the decision tree classifier by building the tree structure
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y, depth=0): # using Gini Index for recursively building the tree
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            return {"class": unique_classes[0]}
        
        if self.max_depth and depth >= self.max_depth or num_samples < self.min_samples_split:
            return {"class": self._majority_class(y)}
        
        best_split = self._best_split(X, y)
        
        left_tree = self._build_tree(X[best_split["left_indices"]], y[best_split["left_indices"]], depth + 1)
        right_tree = self._build_tree(X[best_split["right_indices"]], y[best_split["right_indices"]], depth + 1)
        
        return {
            "feature": best_split["feature"],
            "threshold": best_split["threshold"],
            "left": left_tree,
            "right": right_tree
        }

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_split = {}
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])           
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                left_y = y[left_indices]
                right_y = y[right_indices]
                if len(left_y) < self.min_samples_split or len(right_y) < self.min_samples_split:
                    continue
                
                gini = self._gini_impurity(left_y, right_y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_split = {
                        "feature": feature_idx,
                        "threshold": threshold,
                        "left_indices": left_indices,
                        "right_indices": right_indices
                    }
        
        return best_split
    
# Calculating the gini impurity for the split
    def _gini_impurity(self, left_y, right_y):
        left_size = len(left_y)
        right_size = len(right_y)
        total_size = left_size + right_size       
        left_impurity = 1 - sum([(np.sum(left_y == c) / left_size) ** 2 for c in np.unique(left_y)])
        right_impurity = 1 - sum([(np.sum(right_y == c) / right_size) ** 2 for c in np.unique(right_y)])       
        return (left_size / total_size) * left_impurity + (right_size / total_size) * right_impurity
    
    def _majority_class(self, y):
        return np.bincount(y).argmax()

class XGBoostClassification:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=50):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None

# Training the model with gradient boosting with decision tree
    def fit(self, X, y):
        np.random.seed(self.random_state)
        y_pred = np.zeros_like(y, dtype=np.float64)
        self.feature_importances_ = np.zeros(X.shape[1])
        for i in range(self.n_estimators):
            residuals = self.compute_gradients(y, y_pred)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X, residuals) 
            y_pred += self.learning_rate * tree.predict(X) 
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_
        self.feature_importances_ /= self.n_estimators

# Computing gradient for logistic loss function
    def compute_gradients(self, y, y_pred):
        probabilities = self.sigmoid(y_pred)
        gradients = probabilities - y 
        return gradients

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X) 
        probabilities = self.sigmoid(y_pred)
        return (probabilities <= 0.5).astype(int) 

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

