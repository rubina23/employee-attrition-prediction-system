# Employee Attrition Prediction System

## 📌 Project Overview
This project predicts employee attrition (whether an employee is likely to leave or stay) using **Machine Learning**.  
It includes:
- End‑to‑end ML workflow (data preprocessing → training → evaluation → deployment).
- A **Gradio web app** for user‑friendly predictions.
- Deployment ready for **Hugging Face Spaces**.

---

## ⚙️ Tech Stack
- **Python** (pandas, numpy, scikit‑learn)
- **Machine Learning Models**: Random Forest Classifier
- **Gradio** (web interface)
- **Pickle** (model persistence)
- **Hugging Face Spaces** (deployment)

---

## 📂 Repository Structure

```
├── employee_train.py               # Training pipeline (data prep, model training, evaluation, saving)
├── app.py                          # Gradio web interface for predictions
├── employee_model.pkl              # Saved trained model
├── requirement.txt                 # Dependencies
├── README.md                       # Project documentation
└── Employee-Attrition.csv          # Dataset

```


---

## 🚀 Steps Implemented
1. **Data Loading** – Load HR dataset and verify shape.
2. **Preprocessing** – Handle missing values, encode target, scale numeric features, one‑hot encode categorical features.
3. **Pipeline Creation** – Integrated preprocessing + model.
4. **Model Selection** – Random Forest chosen for robustness and interpretability.
5. **Training** – Train/test split and model fitting.
6. **Cross‑Validation** – 5‑fold CV for robustness.
7. **Hyperparameter Tuning** – GridSearchCV for best parameters.
8. **Evaluation** – Accuracy, classification report, confusion matrix.
9. **Model Saving** – Save pipeline with pickle.
10. **Web Interface** – Gradio app for interactive predictions.
11. **Deployment** – Hugging Face Spaces ready.

---

## 🖥️ Gradio App Usage
Run locally:
```bash
python app.py

```

# 📊 Example Predictions



---

# 🌐 Deployment
To deploy on Hugging Face Spaces:

1. Push repo to GitHub.

2. Connect Hugging Face account → create new Space.

3. Select Gradio as SDK.

4. Upload files (app.py, employee_model.pkl, requirement.txt).

5. Space will auto‑build and launch your app.

----


# 🚀 Live Link

**https://huggingface.co/spaces/rubina25/Employee-Attrition-Prediction-System**

Check out the deployed app here: [Employee Attrition Prediction System](https://huggingface.co/spaces/rubina25/Employee-Attrition-Prediction-System) 🌐
