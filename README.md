-----

# 🚀 AI-Powered SAP Ticket Classifier

This project leverages a deep learning model to automatically classify SAP support tickets. By analyzing the **subject** and **content** of a ticket, it predicts two crucial fields:

  * **Module**: e.g., SD, MM, HCM, FICO
  * **Request Type**: e.g., Incident, Service Request, CR Modification

-----

## 📁 Project Structure

```
📁 AiModel/
├── ai_ticket_classifier.py # Main script for training and prediction
├── ticket_data.csv         # Training data file
└── README.md               # Project documentation
```

-----

## 🛠️ Installation

Ensure you have **Python 3.10+** installed.

Install the necessary Python packages using pip:

```bash
pip install pandas scikit-learn tensorflow
```

-----

## 📊 Input Dataset (`ticket_data.csv`)

The `ticket_data.csv` file serves as the training data and must contain at least these four columns:

| `subject`                  | `content`                                   | `module` | `request_type`  |
| :------------------------- | :------------------------------------------ | :------- | :-------------- |
| INC-12345 PRICE ISSUE      | We are unable to update pricing condition... | SD       | Incident        |
| CR-009 Custom Report       | We need a new report for Material Movement... | MM       | CR Modification |

You can add more rows to this file to enhance the model's training accuracy.

-----

## ▶️ How to Run

### Train the Model

Run the training step initially or whenever you update the `ticket_data.csv` file. This will train the model and save it.

```bash
python ai_ticket_classifier.py
```

### Predict a New Ticket

To test the model with new input, modify the `ai_ticket_classifier.py` script by updating the `test_subject` and `test_content` variables within the prediction section:

```python
test_subject = "Employee Master Data Not Syncing"
test_content = """
Employee details entered in PA30 are not reflecting in the ESS portal.
The issue started after last weekend's maintenance activity.
"""

module, req_type = predict(test_subject, test_content)
print("Predicted Module:", module)
print("Predicted Request Type:", req_type)
```

#### Sample Output:

```
Predicted Module: HCM
Predicted Request Type: Incident
```

-----

## 🧠 Model Overview

The system employs a **Tokenizer** for text preprocessing, which then feeds the sequence into a **Keras neural network**.

The project uses two distinct models:

  * `module_model.h5`: Predicts the SAP module.
  * `request_type_model.h5`: Predicts the ticket request type.

-----

## 💡 Common Errors

  * **`ValueError` during `train_test_split`**: This usually indicates your dataset is too small. Add more rows to `ticket_data.csv`.
  * **Tokenizer error**: Ensure you have run the training step (by executing `python ai_ticket_classifier.py`) before attempting to use the prediction functionality.

-----

## 🌱 Future Enhancements

  * **Improved NLP Accuracy**: Explore replacing the current feedforward neural network with more advanced models like **BERT** or **LSTM** for better natural language processing.
  * **Real-time Predictions**: Deploy the model using frameworks like **Flask** or **FastAPI** to enable real-time predictions.
  * **User Interface**: Develop a simple web UI using technologies such as **React** or **Streamlit** for easier interaction.

-----

## 👨‍💻 Author

Suraj Vishwakarma
