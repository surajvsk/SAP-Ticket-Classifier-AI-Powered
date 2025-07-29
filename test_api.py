import requests
import json

def test_api():
    base_url = "http://127.0.0.1:8000"
    
    # Test health check
    print("=== Testing Health Check ===")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Health check failed: {e}")
    
    print("\n=== Testing Labels Endpoint ===")
    try:
        response = requests.get(f"{base_url}/labels")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Labels endpoint failed: {e}")
    
    # Test prediction with your exact payload
    print("\n=== Testing Prediction ===")
    test_data = {
        "subject": "Correction in report ZFIR14",
        "content": "As per our discussion on MST yesterday, please correct..."
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Prediction failed: {e}")
    
    # Test with another example
    print("\n=== Testing Another Example ===")
    test_data2 = {
        "subject": "New Employee ID Creation Request",
        "content": "Please create a new employee ID for Mr. Ramesh Kumar who joined on 25th July. Assign to Mumbai cost center 2023."
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data2)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_api()