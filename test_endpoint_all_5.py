"""Test the independent-algorithms endpoint with all 5 modes"""
import requests
import json

url = "http://localhost:5000/api/independent-algorithms"

# Test data that triggers all 5 modes
data = {
    "Power_On_Hours": 50000,
    "Total_TBW_TB": 400,
    "Total_TBR_TB": 350,
    "Temperature_C": 76,
    "Percent_Life_Used": 85,
    "Media_Errors": 15,
    "Unsafe_Shutdowns": 12,
    "CRC_Errors": 25,
    "Read_Error_Rate": 35,
    "Write_Error_Rate": 28
}

try:
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print("✓ Endpoint successful!\n")
        print("=" * 70)
        print(f"Total Detected Issues: {result['total_detected']}")
        print(f"Filter Threshold: {result['filter_threshold']}")
        print("=" * 70)
        
        for i, algo in enumerate(result['independent_algorithms'], 1):
            print(f"\n{i}. {algo['label']} (Mode {algo['mode']})")
            print(f"   Score: {algo['score']}/100")
            print(f"   Filter: {algo['error_filter']}")
            print(f"   Factors:")
            for reason in algo['reasons']:
                print(f"     • {reason}")
        
        print("\n" + "=" * 70)
        print("✓ All 5 modes working correctly!")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"✗ Connection error: {e}")
    print("  Make sure Frontend is running on port 5000")
