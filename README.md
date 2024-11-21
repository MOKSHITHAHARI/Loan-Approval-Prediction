# Loan-Approval-Prediction
# LoanSure  
**A Machine Learning-Based Loan Approval Prediction System with Explainability and Privacy.**  

## Abstract  
LoanSure is a project focused on enhancing the loan approval process using machine learning. With a dataset of 614 applicants containing attributes such as dependents, education, applicant income, loan amount, credit history, and property area, this project aims to predict loan outcomes while addressing key challenges in explainability and data privacy.  

While machine learning models excel at prediction, they often lack transparency and pose risks to sensitive data. LoanSure tackles these issues by integrating explainability tools like **SHAP** (SHapley Additive exPlanations) and **LIME** (Local Interpretable Model-agnostic Explanations), ensuring stakeholders understand the factors influencing decisions. Additionally, **differential privacy** safeguards sensitive financial information during model training.  

## Project Objectives  
The primary objectives of this project are:  
1. **Develop a Robust Loan Approval Model:** Implement machine learning classification techniques for accurate predictions.  
2. **Ensure Explainability:** Leverage SHAP and LIME to make predictions transparent and interpretable for financial analysts and applicants.  
3. **Enhance Data Privacy:** Incorporate privacy-preserving techniques like differential privacy to protect sensitive information against unauthorized exposure.  

## Expected Results  
1. **Accurate Loan Predictions:** Achieving high performance metrics, including accuracy, precision, recall, and F1 score.  
2. **Transparent Decision-Making:** Clear, interpretable explanations for loan approvals or rejections using SHAP and LIME.  
3. **Privacy Protection:** Secure borrower data with differential privacy, maintaining compliance with regulations like GDPR.  

## Dataset  
The project uses the `loan-approval-prediction.csv` dataset, containing 614 rows of historical loan application data. Key features include:  
- **Dependents**  
- **Education**  
- **Self Employment**  
- **Applicant Income**  
- **Co-Applicant Income**  
- **Loan Amount**  
- **Loan Duration**  
- **Credit History**  
- **Property Area**  

## Key Features  
1. **Explainability with SHAP and LIME:**  
   - Provide detailed insights into the factors driving each loan prediction.  
   - Build trust and accountability in the decision-making process.  

2. **Privacy-Preserving Techniques:**  
   - Use differential privacy to anonymize sensitive information while maintaining data utility.  

3. **Interactive Streamlit Dashboard:**  
   - Visualize loan decisions dynamically based on user input.  
   - Enable financial analysts to evaluate profiles interactively.  

## Technical Details  
- **Language:** Python  
- **Libraries:** Scikit-learn, Pandas, NumPy, SHAP, LIME, Streamlit  
- **Privacy Frameworks:** Differential Privacy Toolkit  
- **Machine Learning Models:** Classification Algorithms (e.g., Logistic Regression, Decision Trees, Random Forest)  
