
## **10. Web Interface with Gradio (10 Marks)**

# Create a user-friendly Gradio web interface that takes user inputs and displays the prediction from your trained model.


import pandas as pd
import gradio as gr
import pickle


# 1. Loaded saved model
with open("employee_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# 2. Prediction function
def predict_attrition(Age, MonthlyIncome, JobRole, OverTime):
    input_data = pd.DataFrame([[Age, MonthlyIncome, JobRole, OverTime]],
                              columns=['Age','MonthlyIncome','JobRole','OverTime'])
    prediction = pipeline.predict(input_data)[0]
    return "Attrition" if prediction == 1 else "No Attrition"

# 3. Gradio interface
inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Monthly Income"),
    gr.Dropdown(label="Job Role", choices=[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]),
    gr.Dropdown(label="OverTime", choices=["Yes", "No"])
]

employee_app = gr.Interface(
    fn=predict_attrition,
    inputs=inputs,
    outputs="text",
    title="Employee Attrition Prediction",
    description="Enter employee details to predict attrition."
)

employee_app.launch(share=True)

"""## **11. Deployment to Hugging Face (10 Marks)**
Hugging Face Spaces public URL:
"""