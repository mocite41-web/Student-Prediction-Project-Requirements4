# Student Performance Prediction System

A classroom prototype that uses historical student information to estimate whether a student may be at risk of failing the final Mathematics grade.

## Prediction rule

- `G3 < 10` = **AT RISK**
- `G3 >= 10` = **NOT AT RISK**

`G3` is the target only. It is removed from the model's input features to prevent target leakage.

## Dataset

This project retrieves the UCI Student Performance dataset (dataset ID 320) using `ucimlrepo`. The laboratory exercise specifies the Mathematics records. The training script saves the retrieved records as `student_data.csv`.

Dataset page: https://archive.ics.uci.edu/dataset/320/student+performance

## Technologies

- Python
- pandas
- scikit-learn
- Random Forest classifier
- Streamlit
- joblib
- GitHub

## Installation

Use Python 3.10 or later. Open a terminal in this project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

## Train and evaluate the model

```bash
python train_model.py
```

This creates:
- `student_data.csv`
- `student_model.pkl`
- `evaluation_results.json`

The script prints accuracy, precision, recall, F1-score, and a classification report. Use the actual values printed by your run in your submission; results are not filled in here because they must come from executing the model.

## Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, normally http://localhost:8501. Download the sample input CSV from the dashboard, then upload it to test predictions.

## System workflow

1. Retrieve and inspect the dataset.
2. Remove duplicate records.
3. Create the binary target from `G3`.
4. Exclude `G3` from predictor columns.
5. Impute missing values, scale numeric columns, and encode categorical columns.
6. Train and evaluate a Random Forest classifier.
7. Save the model and display predictions through Streamlit.

## Limitations and responsible use

This is a classroom demonstration using historical data. The model was trained using one dataset from Portuguese secondary schools, so it may not generalize to students at another institution or population. Predictions are not diagnoses and must not be used as the sole basis for real student decisions. A faculty member should review the student's situation and provide appropriate support.

## GitHub

Create a public repository named `student-performance-prediction`, then run these commands after replacing `YOUR-USERNAME`:

```bash
git init
git add train_model.py app.py requirements.txt README.md
git commit -m "Build student performance prediction prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/student-performance-prediction.git
git push -u origin main
```
