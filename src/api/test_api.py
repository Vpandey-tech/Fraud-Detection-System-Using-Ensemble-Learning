"""
Simple Test Suite for Fraud Detection API
Run this after starting the API server to verify everything works.
"""

import requests
import json

API_URL = "http://localhost:8000"

def test_api_health():
    """Test if API is running"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{API_URL}/")
        if response.status_code == 200:
            print("✅ API is running!")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        print("   Make sure you started the server with: uvicorn src.api.main:app --reload")
        return False

def test_normal_transaction():
    """Test a normal (non-fraud) transaction"""
    print("\n🔍 Testing Normal Transaction...")
    
    # Create a normal transaction (all features near 0)
    transaction = {f'v{i}': 0.0 for i in range(1, 29)}
    transaction['scaled_amount'] = 0.5  # Small amount
    
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction)
        result = response.json()
        
        print(f"   Fraud Score: {result['fraud_score']:.4f}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Explanation: {result['explanation']}")
        print(f"   Processing Time: {result.get('processing_time_ms', 'N/A')}ms")
        
        if not result['is_fraud']:
            print("✅ Normal transaction correctly identified!")
            return True
        else:
            print("⚠️ Warning: Normal transaction flagged as fraud")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_fraud_transaction():
    """Test a fraudulent transaction"""
    print("\n🔍 Testing Fraud Transaction...")
    
    # Create a fraud transaction with known fraud patterns
    transaction = {f'v{i}': 0.0 for i in range(1, 29)}
    transaction['v4'] = 5.0   # High fraud indicator
    transaction['v14'] = -3.0  # High fraud indicator
    transaction['scaled_amount'] = 0.5
    
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction)
        result = response.json()
        
        print(f"   Fraud Score: {result['fraud_score']:.4f}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Explanation: {result['explanation']}")
        print(f"   Processing Time: {result.get('processing_time_ms', 'N/A')}ms")
        
        if result['is_fraud']:
            print("✅ Fraud transaction correctly detected!")
            return True
        else:
            print("⚠️ Warning: Fraud transaction not detected")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_high_value_transaction():
    """Test high-value transaction (should trigger hybrid rule)"""
    print("\n🔍 Testing High-Value Transaction...")
    
    transaction = {f'v{i}': 0.0 for i in range(1, 29)}
    transaction['scaled_amount'] = 25.0  # Very high amount (>$2000)
    
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction)
        result = response.json()
        
        print(f"   Fraud Score: {result['fraud_score']:.4f}")
        print(f"   Is Fraud: {result['is_fraud']}")
        print(f"   Explanation: {result['explanation']}")
        
        if result['fraud_score'] >= 0.95:
            print("✅ High-value rule correctly triggered!")
            return True
        else:
            print("⚠️ Warning: High-value rule not triggered")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_invalid_input():
    """Test that invalid input is rejected"""
    print("\n🔍 Testing Input Validation...")
    
    # Send invalid data (missing fields)
    invalid_transaction = {'v1': 0.0, 'v2': 0.0}  # Missing most fields
    
    try:
        response = requests.post(f"{API_URL}/predict", json=invalid_transaction)
        
        if response.status_code == 422:  # Validation error
            print("✅ Invalid input correctly rejected!")
            return True
        else:
            print(f"⚠️ Expected validation error, got status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("🚀 FRAUD DETECTION SYSTEM - TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_api_health,
        test_normal_transaction,
        test_fraud_transaction,
        test_high_value_transaction,
        test_invalid_input
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Your system is working perfectly!")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    run_all_tests()
