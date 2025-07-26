# 🚀 SAP Ticket Classifier (AI-Powered)

This project uses a deep learning model to automatically classify SAP support tickets based on their **subject** and **content**. It predicts two key fields:

- 🔧 **Module**: e.g., SD, MM, HCM, FICO
- 📝 **Request Type**: e.g., Incident, Service Request, CR Modification

---

## 📁 Files in the Project

📁 AiModel/
├── ai_ticket_classifier.py # Main training + prediction script
├── ticket_data.csv # Training data file
└── README.md # Documentation

---

## 🛠️ Installation

Make sure Python 3.10+ is installed.

Install required Python packages:

```bash
pip install pandas scikit-learn tensorflow
📊 Input Dataset (ticket_data.csv)
The file should contain at least these four columns:

subject	content	module	request_type
INC-12345 PRICE ISSUE	We are unable to update pricing condition...	SD	Incident
CR-009 Custom Report	We need a new report for Material Movement...	MM	CR Modification

You can add more rows to improve training accuracy.

▶️ How to Run
Train the model (first time or when data is updated):

python ai_ticket_classifier.py
Predict a new ticket

Inside the same script (ai_ticket_classifier.py), modify this section to test new input:


test_subject = "Employee Master Data Not Syncing"
test_content = """
Employee details entered in PA30 are not reflecting in the ESS portal.
The issue started after last weekend's maintenance activity.
"""

module, req_type = predict(test_subject, test_content)
print("Predicted Module:", module)
print("Predicted Request Type:", req_type)
Sample Output:


Predicted Module: HCM
Predicted Request Type: Incident
🧠 Model Overview
Uses Tokenizer to preprocess text input.

Feeds the sequence into a Keras neural network.

Two models:

module_model.h5 – predicts SAP module

request_type_model.h5 – predicts ticket type

💡 Common Errors
ValueError during train_test_split: Your dataset is too small. Add more rows.

Tokenizer error: Run the training step before using prediction.

🌱 Future Scope
Replace feedforward NN with BERT or LSTM for better NLP accuracy.

Deploy using Flask / FastAPI for real-time predictions.

Add a simple web UI using React or Streamlit.

👨‍💻 Author
Name: Dynamite Technology
