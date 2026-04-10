"""Test the new independent-algorithms endpoint"""
import requests
import json

url = "http://localhost:5000/api/independent-algorithms"
data = {
    "Power_On_Hours": 50000,
    "Total_TBW_TB": 400,
    "Total_TBR_TB": 350,
    "Temperature_C": 76,
    "Percent_Life_Used": 85,
    "Media_Errors": 15,
    "Unsafe_Shutdowns": 8,
    "CRC_Errors": 25,
    "Read_Error_Rate": 35,
    "Write_Error_Rate": 28
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Endpoint successful!\n")
        print(json.dumps(result, indent=2))
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"✗ Connection error: {e}")
    print("  Make sure Frontend is running on port 5000")
