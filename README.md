# 🏦 Bank Customer Churn Prediction using Deep Learning (ANN)

A comprehensive end-to-end machine learning project that predicts customer churn in banks using Artificial Neural Networks (ANN). This solution enables banks to identify at-risk customers and implement targeted retention strategies.

---

## 📌 Problem Statement

Customer churn is a major challenge for banks, as acquiring new customers is significantly more expensive than retaining existing ones. This project aims to **predict whether a customer will leave the bank (churn) based on their demographic and account-related information**, enabling proactive customer retention interventions.

---

## 🎯 Project Objectives

- Build an end-to-end Deep Learning model to predict customer churn with high accuracy
- Identify key patterns and features influencing customer churn behavior
- Improve prediction performance using ANN with advanced regularization techniques
- Enable data-driven decision-making for customer retention strategies
- Deploy an interactive web application for real-time churn predictions

---

## 📊 Dataset Overview

- **Source**: Kaggle – Bank Customer Churn Dataset
- **Total Records**: ~10,000 customers
- **Total Features**: 10 input variables
- **Target Variable**: `exited` (1 = Churn, 0 = No Churn)
- **Class Imbalance**: 20% churn rate (2,000 customers)

---

## 📖 Data Dictionary

| Feature Name | Description | Type |
|---|---|---|
| `credit_score` | Customer's credit score | Numerical |
| `geography` | Customer's country (France, Germany, Spain) | Categorical |
| `gender` | Gender of the customer (Male/Female) | Categorical |
| `age` | Age of the customer | Numerical |
| `tenure` | Number of years with the bank | Numerical |
| `balance` | Account balance | Numerical |
| `num_of_products` | Number of bank products used | Numerical |
| `has_cr_card` | Credit card ownership (1=Yes, 0=No) | Binary |
| `is_active_member` | Active member status (1=Yes, 0=No) | Binary |
| `estimated_salary` | Estimated yearly salary | Numerical |
| `exited` | Churn status (1=Yes, 0=No) | Target |

---

## 🛠️ Tech Stack

### Programming & Data Processing
- **Python 3.10+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations

### Visualization
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations

### Machine Learning & Preprocessing
- **Scikit-learn** - Preprocessing, scaling, encoding
- **Imbalanced-learn** - SMOTE for handling class imbalance

### Deep Learning
- **TensorFlow / Keras** - Neural network implementation
- **KerasTuner** - Hyperparameter optimization

### Deployment
- **Streamlit** - Interactive web application
- **Joblib** - Model serialization

---

## 🔄 Project Workflow

### 1. **Data Collection & Loading**
   - Downloaded dataset from Kaggle
   - Loaded into Python environment with Pandas

### 2. **Data Understanding & Cleaning**
   - Analyzed data structure and types
   - Handled missing values
   - Removed irrelevant features (customer_id)
   - Data validation and quality checks

### 3. **Exploratory Data Analysis (EDA)**
   - Analyzed feature distributions
   - Studied relationships between features and churn
   - Identified patterns using statistical analysis
   - Created visualizations for insights
   - Discovered age, tenure, and geography as key factors

### 4. **Feature Engineering**
   - **Balance per Product**: `balance / (num_of_products + 1)` - Measures account value per service
   - **Engagement Score**: `num_of_products × is_active_member` - Captures customer activity
   - **Age Groups**: Binned into [18-30], [30-45], [45-60], [60+] - Demographic segmentation
   - **Zero Balance Flag**: Binary indicator for inactive accounts

### 5. **Data Preprocessing**
   - **Handling Categorical Variables**: One-hot encoding with drop_first=True
   - **Feature Scaling**: StandardScaler for numerical features
   - **Encoding Strategy**:
     - 8 Numerical features → Scaled
     - 6 Categorical features → One-hot encoded
     - 3 Binary features → Passthrough
   - **Final Feature Count**: 17 features

### 6. **Model Building (ANN)**

#### Architecture
```
Input Layer (17 features)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Output Layer (1 unit, Sigmoid activation)
```

#### Key Specifications
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, AUC-ROC
- **Batch Size**: 32
- **Epochs**: 100 (with Early Stopping)

### 7. **Model Optimization & Regularization**
- **Dropout**: Implemented dropout layers to prevent overfitting
- **Early Stopping**: Monitored validation loss with patience=20
- **Model Checkpoint**: Saved best model during training
- **Class Weights**: Balanced for handling class imbalance (20% churn)
- **Learning Rate**: Adaptive learning rate with Adam optimizer

### 8. **Hyperparameter Tuning**
- Used **KerasTuner** with RandomSearch strategy
- Tuned: Hidden units, dropout rate, learning rate
- Evaluated 10 different configurations
- Selected best model based on recall metric

### 9. **Class Imbalance Handling**
- **Method 1**: Class weight balancing (selected)
- **Method 2**: SMOTE oversampling (tested)
- **Method 3**: Threshold tuning at 0.35 (for better recall)

### 10. **Model Evaluation**

#### Performance Metrics
| Model | Accuracy | Recall | Precision | AUC-ROC |
|---|---|---|---|---|
| Baseline | 87% | 48% | - | 0.86 |
| Threshold Tuned | 85% | 67% | - | 0.85 |
| Class Weighted | 80% | 73% | 51% | 0.82 |
| SMOTE | 82% | 68% | - | 0.80 |
| Hyperparameter Tuned | 86% | 43% | - | 0.86 |
| **Final Optimized** | **80%** | **72%** | **51%** | **0.84** |

#### Why Recall is Prioritized
- Missing a churner (False Negative) = Revenue loss
- Falsely predicting churn (False Positive) = Unnecessary retention cost
- 72% recall captures ~7 out of 10 actual churners
- Enables cost-effective, targeted retention interventions

### 12. **Model Comparison & Selection**
- Compared baseline, regularized, and tuned models
- Selected final model optimized for recall
- Achieved best balance between capturing churners and false positives

---

## 📁 Project Structure

```
bank-customer-churn-prediction/
│
├── 📄 01_project_overview.ipynb            # Project introduction & objectives
├── 📄 02_data_understanding_cleaning.ipynb # Data loading, cleaning, validation
├── 📄 03_eda.ipynb                         # Exploratory data analysis
├── 📄 04_feature_engineering.ipynb         # Feature creation & transformation
├── 📄 05_modelling.ipynb                   # ANN model building & optimization
├── 📄 06_app.py                            # Streamlit web application
│
├── data/
│   ├── churn Modeling.csv                  # Original dataset
│   ├── final_data.csv                      
    ├── cleaned_data.csv                    
│   └── df.csv                              
│
├── churn_model.keras                 # Trained ANN model
├── preprocessor.pkl                  # Fitted preprocessor
│
├── requirements.txt                      # Project dependencies
├── README.md                             # Project documentation
└── .gitignore                            # Git ignore file
```

---

## 🚀 How to Use

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/SagarSingh2004/bank-churn-prediction.git
cd bank-churn-prediction
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Application

#### Dashboard Page
- View overall model performance metrics
- Explore feature importance analysis
- Understand feature engineering techniques

#### Predict Churn Page
1. Enter customer details:
   - Personal Information (Age, Gender, Country)
   - Account Details (Credit Score, Tenure)
   - Financial Information (Balance, Salary)
   - Product & Membership (Products, Credit Card, Active Status)
2. Click "Predict Churn Risk"
3. View:
   - Churn probability (0-100%)
   - Binary prediction (0=Stay, 1=Churn)
   - Risk gauge visualization
   - Personalized recommendations

#### Analytics Page
- Explore churn distribution by:
  - Age group patterns
  - Country/Geography insights
  - Gender analysis
  - Tenure trends

#### About Model Page
- Detailed model architecture
- Dataset characteristics
- Preprocessing pipeline
- Feature engineering details
- Performance metrics explanation
- Technical stack overview

---

## 📊 Key Findings

### Features Most Impacting Churn
1. **Age** (28%) - Older customers have higher churn rates
2. **Engagement Score** (22%) - Active members are more likely to stay
3. **Tenure** (18%) - Longer tenure reduces churn risk
4. **Number of Products** (15%) - More products = higher retention
5. **Active Membership** (12%) - Critical retention indicator
6. **Credit Score** (5%) - Less influential but still relevant

### Churn Patterns Discovered
- **Age Groups**: 60+ has 37% churn vs 17% in 18-30 group
- **Geography**: Germany shows 32% churn (highest), France 16% (lowest)
- **Gender**: Female customers have 25% churn vs 16% for males
- **Tenure**: New customers (0-1 year) have 27% churn rate
- **Balance**: Zero balance indicates 3x higher churn risk
- **Inactivity**: Inactive members show significantly higher churn

---

## 💡 Business Recommendations

### For High-Risk Customers
- 🎁 Offer exclusive loyalty rewards and discounts
- 📞 Initiate proactive engagement calls
- 💰 Promote complementary products and services
- 🚀 Launch re-engagement campaigns for inactive members

### For New Customers
- 📱 Provide onboarding support and education
- 🎯 Customize product recommendations
- 📊 Regular check-ins to ensure satisfaction

### For Low-Tenure Customers
- 🏆 Incentivize multi-product adoption
- 💳 Cross-sell complementary financial products
- 📈 Showcase account growth and benefits

---

## 📈 Model Performance Insights

### Strengths
✅ 72% Recall - Captures majority of at-risk customers  
✅ 80% Accuracy - High overall correctness  
✅ 0.84 AUC-ROC - Excellent discrimination ability  
✅ Balanced approach - Considers business costs  

### Areas for Future Improvement
🔄 Collect additional behavioral data (transaction patterns, communication history)  
🔄 Implement ensemble methods (combining multiple models)  
🔄 Explore LSTM/RNN for temporal patterns  
🔄 Add customer interaction and satisfaction metrics  

---

## 🛠️ Requirements

```
pandas==1.5.0
numpy==1.23.0
scikit-learn==1.1.0
tensorflow==2.11.0
keras==2.11.0
keras-tuner==1.1.0
imbalanced-learn==0.9.0
matplotlib==3.6.0
seaborn==0.12.0
plotly==5.11.0
streamlit==1.15.0
joblib==1.1.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📧 Contact & Support

For questions, suggestions, or issues:
- Create an issue on GitHub
- Contact: [itsofficialworkforss.com]
- LinkedIn: [www.linkedin.com/in/sagar-singh-10a53028a]

---

## 🙏 Acknowledgments

- **Dataset Source**: [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/churn-dataset)
- **Inspiration**: Banking industry best practices for customer retention
- **Tools**: TensorFlow, Keras, Scikit-learn, Streamlit communities

---

## 🌟 Project Highlights

- ⭐ **End-to-End ML Pipeline**: From raw data to production-ready deployment
- ⭐ **Deep Learning Focus**: ANN with advanced optimization techniques
- ⭐ **Interactive UI**: User-friendly Streamlit application
- ⭐ **Business-Ready**: Actionable insights and recommendations
- ⭐ **Well-Documented**: Comprehensive Jupyter notebooks and documentation
- ⭐ **Production-Grade**: Model serialization and preprocessing pipelines

---

## 📚 Learning Resources

### Concepts Covered
- Binary Classification with Deep Learning
- Handling Class Imbalance
- Feature Engineering & Domain Knowledge
- Model Optimization & Regularization
- Hyperparameter Tuning with KerasTuner
- Model Evaluation & Metric Selection
- Streamlit Web Application Development

### Recommended Reading
- [Deep Learning for Binary Classification](https://www.deeplearningbook.org/)
- [Imbalanced Classification Guide](https://machinelearningmastery.com/imbalanced-classification-using-smote/)
- [Feature Engineering Best Practices](https://www.coursera.org/learn/machine-learning-engineering-production)

---

**Made with ❤️ for Banking Analytics & ML Practitioners**

Last Updated: 2026 | Version: 1.0.0