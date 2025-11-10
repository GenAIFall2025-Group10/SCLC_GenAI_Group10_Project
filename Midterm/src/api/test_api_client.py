"""
OncoDetect-AI: Test API Client
Test your Risk Score API endpoints
"""

import requests
import json


class RiskScoreAPIClient:
    """Client to interact with Risk Score API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self):
        """Check if API is running"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def get_model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()
    
    def predict_risk(self, patient_data):
        """Predict risk for a single patient"""
        response = requests.post(
            f"{self.base_url}/predict",
            json=patient_data
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    
    def predict_batch(self, patients_list):
        """Predict risk for multiple patients"""
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json=patients_list
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None


def test_api():
    """Test the API with sample patients"""
    
    print("=" * 70)
    print("🧪 Testing OncoDetect-AI Risk Score API")
    print("=" * 70)
    
    # Initialize client
    client = RiskScoreAPIClient()
    
    # Test 1: Health check
    print("\n1️⃣ Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    # Test 2: Model info
    print("\n2️⃣ Model Information:")
    model_info = client.get_model_info()
    print(json.dumps(model_info, indent=2))
    
    # Test 3: Single patient prediction - HIGH RISK profile
    print("\n3️⃣ High Risk Patient Prediction:")
    high_risk_patient = {
        "age": 75,
        "is_male": 1,
        "is_former_smoker": 0,
        "mutation_count": 450,
        "tmb": 18.5,
        "is_tmb_intermediate": 1,
        "is_tmb_low": 0,
        "smoker_x_tmb": 0.0
    }
    
    result = client.predict_risk(high_risk_patient)
    if result:
        print(f"\n📊 Prediction Results:")
        print(f"   Risk Score: {result['risk_score']:.1f}/100")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"\n💡 Interpretation:")
        print(f"   {result['clinical_interpretation']}")
        print(f"\n🏥 Recommendation:")
        print(f"   {result['recommended_action']}")
    
    # Test 4: Single patient prediction - LOW RISK profile
    print("\n4️⃣ Low Risk Patient Prediction:")
    low_risk_patient = {
         "age": 52,
         "is_male": 0,
         "is_former_smoker": 1,
         "mutation_count": 85,
         "tmb": 3.2,
         "is_tmb_intermediate": 0,
         "is_tmb_low": 1,
         "smoker_x_tmb": 0.0
    }
    
    result = client.predict_risk(low_risk_patient)
    if result:
        print(f"\n📊 Prediction Results:")
        print(f"   Risk Score: {result['risk_score']:.1f}/100")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"\n💡 Interpretation:")
        print(f"   {result['clinical_interpretation']}")
        print(f"\n🏥 Recommendation:")
        print(f"   {result['recommended_action']}")
    
    # Test 5: Batch prediction
    print("\n5️⃣ Batch Prediction (2 patients):")
    batch_result = client.predict_batch([high_risk_patient, low_risk_patient])
    if batch_result:
        print(f"✓ Processed {batch_result['count']} patients")
        for i, pred in enumerate(batch_result['predictions']):
            print(f"\n   Patient {i+1}: Risk Score = {pred['risk_score']:.1f}, Level = {pred['risk_level']}")
    
    print("\n" + "=" * 70)
    print("✅ All API Tests Completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_api()