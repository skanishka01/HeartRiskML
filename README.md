# HeartRiskML
Heart Disease Risk Prediction using Ensemble Learning

This project demonstrates the application of machine learning techniques for classification tasks, with a focus on healthcare-related applications such as heart disease detection.
## *Dataset*
The model was trained using the [Heart Disease Dataset](https://www.kaggle.com/code/kristiannova/heart-disesase-try?select=heart.csv) from the UCI Machine Learning Repository.  
- *Features*:
  - Age, gender, chest pain type, blood pressure, cholesterol levels, etc.
- *Target*: Binary classification of heart disease risk (High/Low).  

By analyzing critical features , the project leverages several machine learning algorithms, including:

- Neural Network (NN)  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Logistic Regression  
- Random Forest  

## Key Features
Data Preprocessing: Robust preprocessing techniques and outlier handling were implemented to improve data quality and model performance.
Ensemble Learning: An ensemble approach was employed to combine the strengths of individual models, resulting in improved predictive accuracy.

## *Project Structure*

HeartDiseasePrediction/
├── dataset/
│   └── heart.csv
├── models/
│   ├── nn.keras
│   ├── svm.pkl
│   ├── knn.pkl
│   ├── randomForest.pkl
│   ├── logistic.pkl
│   └── ensemble.pkl
├── static/
│   └── style.css
├── templates/
│   ├── form.html
│   ├── result.html
├── app.py
├── README.md
└── requirements.txt

## *Installation*
1. Clone the repository:
   bash
   git clone https://github.com/yourusername/HeartDiseasePrediction.git
   cd HeartDiseasePrediction
   
2. Install dependencies:
   bash
   pip install -r requirements.txt
   
3. Start the Flask server:
   bash
   python app.py
   
4. Open http://127.0.0.1:5000 in your browser.


## Results
Random Forest achieved promising metrics individually.
Ensemble Model outperformed all individual models, achieving an accuracy of 91.22%, Precision : 90%, Recall: 88% ,F1 Score: 89%  
## Conclusion
This project highlights the importance of robust preprocessing and ensemble strategies in building reliable predictive systems. The synergy of diverse machine learning models proves especially effective for healthcare applications, providing valuable insights for heart disease prediction.


Demo of this project:-
![Screenshot 2024-12-11 001004](https://github.com/user-attachments/assets/7b35d127-4df7-4ef0-8623-b52722073391)
![Screenshot 2024-12-11 002043](https://github.com/user-attachments/assets/382f950c-ef42-404d-8e6f-7a07d3babcfc)
![image](https://github.com/user-attachments/assets/1f522c36-844f-434d-8c8b-b8c862189cd0)
![image](https://github.com/user-attachments/assets/164d0eb5-2dca-4039-9c18-9e9ab5074141)


