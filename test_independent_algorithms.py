"""Test independent algorithms with error filtering"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Frontend'))

from ml_pipeline import run_independent_algorithms

# Test data with high temperature and CRC errors
metrics = {
    "Power_On_Hours": 50000,
    "Total_TBW_TB": 400,
    "Total_TBR_TB": 350,
    "Temperature_C": 76,  # High temp (>75°C)
    "Percent_Life_Used": 85,
    "Media_Errors": 15,   # Some errors
    "Unsafe_Shutdowns": 8,  # Multiple shutdowns (>5)
    "CRC_Errors": 25,     # CRC errors
    "Read_Error_Rate": 35,  # >30%
    "Write_Error_Rate": 28,  # Close to 30%
}

print("=" * 70)
print("INDEPENDENT ALGORITHMS TEST")
print("=" * 70)

results = run_independent_algorithms(metrics)

if not results:
    print("\n✓ No significant issues detected (all scores = 0)")
else:
    print(f"\n✓ Found {len(results)} significant issues:\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['label']} (Mode {r['mode']})")
        print(f"   Score: {r['score']}/100")
        print(f"   Filter: {r.get('error_filter', 'N/A')}")
        print(f"   Factors detected:")
        for reason in r['reasons']:
            print(f"     → {reason}")
        print()
