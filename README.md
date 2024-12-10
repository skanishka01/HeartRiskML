# HeartRiskML
Heart Disease Risk Prediction using Ensemble Learning

This project demonstrates the application of machine learning techniques for classification tasks, with a focus on healthcare-related applications such as heart disease detection.
## *Dataset*
The model was trained using the [Heart Disease Dataset](https://www.kaggle.com/code/kristiannova/heart-disesase-try?select=heart.csv) from the UCI Machine Learning Repository.  
- *Features*:
  - Age, gender, chest pain type, blood pressure, cholesterol levels, etc.
- *Target*: Binary classification of heart disease risk (High/Low).  

## *Key Features*
Data Preprocessing: Robust preprocessing techniques and outlier handling were implemented to improve data quality and model performance.
Ensemble Learning: An ensemble approach was employed to combine the strengths of individual models, resulting in improved predictive accuracy.

## *Project Structure*
  ![image](https://github.com/user-attachments/assets/a7de8860-5fd9-4011-85b8-364e18b34887)

## *Installation*
1. Clone the repository:
   bash
   git clone https://github.com/skanishka01/HeartRiskML.git
   cd HeartRiskML
   
2. Install dependencies:
   bash
   pip install -r requirements.txt
   
3. Start the Flask server:
   bash
   python app.py
   
4. Open http://127.0.0.1:5500 in your browser.

## *Model Details*
- *Neural Network*:
  - Framework: Keras  
  - Layers: Input, Hidden (3 layers), Output  
- *K-Nearest Neighbors*:
  - Neighbors: 5  
- *Support Vector Machine*:
  - Kernel: Radial Basis Function (RBF)  
- *Logistic Regression*:
  - Regularization: L2  
- *Random Forest*:
  - Trees: 100  
- *Ensemble*:
  - Weighted voting scheme for final prediction.
 
## *Results*
Random Forest achieved promising metrics individually.
Ensemble Model outperformed all individual models, achieving an accuracy of 94%, Precision : 90%, Recall: 92% ,F1 Score: 91%  
## *Conclusion*
This project highlights the importance of robust preprocessing and ensemble strategies in building reliable predictive systems. The synergy of diverse machine learning models proves especially effective for healthcare applications, providing valuable insights for heart disease prediction.

Demo of this project:-
![Screenshot 2024-12-11 001004](https://github.com/user-attachments/assets/7b35d127-4df7-4ef0-8623-b52722073391)
![Screenshot 2024-12-11 002043](https://github.com/user-attachments/assets/382f950c-ef42-404d-8e6f-7a07d3babcfc)
![image](https://github.com/user-attachments/assets/fbe18eab-7e20-4e2d-b379-224d19a5aa0f)
