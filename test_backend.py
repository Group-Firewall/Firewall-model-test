import requests
import json
import time

url = "http://localhost:8000/predict"

payload = {
    "Source_IP": "192.168.1.10",
    "Destination_IP": "10.0.0.5",
    "Port": 80,
    "Request_Type": "HTTP",
    "Protocol": "TCP",
    "Payload_Size": 1024,
    "User_Agent": "Mozilla/5.0",
    "Status": "Success",
    "Scan_Type": "Normal"
}

print(f"Sending request to {url}...")
try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Request failed: {e}")
