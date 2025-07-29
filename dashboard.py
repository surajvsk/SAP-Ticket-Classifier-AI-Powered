import streamlit as st
import requests

# === Configuration ===
API_URL = "http://127.0.0.1:8000/predict"  # Ensure FastAPI is running

# === Streamlit UI ===
st.set_page_config(page_title="SAP Ticket Classifier", layout="centered")
st.title("SAP Ticket Classifier and Summary Generator")

st.markdown("Enter the ticket details below:")

subject = st.text_input("Subject", placeholder="e.g., New Employee ID Creation Request")
content = st.text_area("Content", height=200, placeholder="Please create a new employee ID for Mr. Ramesh Kumar...")

if st.button("Predict"):
    if not subject or not content:
        st.warning("Please fill in both Subject and Content.")
    else:
        with st.spinner("Sending to model..."):
            payload = {"subject": subject, "content": content}
            try:
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                # Display Predictions
                st.subheader("📌 Prediction")
                st.write(f"**Predicted Module:** `{data['predicted_module']}`")
                st.write(f"**Predicted Request Type:** `{data['predicted_request_type']}`")

                # Display Summary
                st.subheader("📝 Summary")
                st.info(data['summery'])

                # Display Probabilities
                st.subheader("📊 Top 3 Module Predictions")
                for mod in data["top_modules"]:
                    st.write(f"- {mod['label']}: {mod['probability']*100:.2f}%")

                st.subheader("📊 Top 3 Request Type Predictions")
                for req in data["top_request_types"]:
                    st.write(f"- {req['label']}: {req['probability']*100:.2f}%")

            except requests.exceptions.RequestException as e:
                st.error(f"API call failed: {e}")
