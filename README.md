📰 Fake News Detection using Machine Learning

This project implements a Fake News Detection system using Python, scikit-learn, and Natural Language Processing (NLP) techniques.
It demonstrates how machine learning can be applied to text classification, specifically distinguishing between real and fake news articles.

✨ Key Highlights

📂 Dataset: Fake and Real News Dataset (Kaggle)

🧹 Preprocessing: stopword removal, punctuation cleaning, lowercase normalization, URL removal

🔠 Feature Extraction: TF-IDF Vectorization

🤖 Models:

Logistic Regression (baseline)

Multinomial Naive Bayes (comparison)

📊 Evaluation: Accuracy, Classification Report, Confusion Matrix

📝 Prediction: Classifies new text as Real or Fake

📂 Project Structure
fake-news-detection/
│
├── src/                        # Source code
│   ├── data_loader.py          # Load + unzip dataset
│   ├── preprocess.py           # Text cleaning functions
│   ├── train.py                # Train and save model
│   ├── evaluate.py             # Evaluate model with confusion matrix
│   └── predict.py              # Predict on new sample texts
│
├── sample_predictions.csv      # Example model outputs
├── requirements.txt            # Dependencies
├── README.md                   # Project description & usage instructions
└── .gitignore                  # Ignore unnecessary files
