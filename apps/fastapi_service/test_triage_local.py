import asyncio
import joblib
import onnxruntime as ort

from src.shared.config import ServiceSettings
from src.triage.feature_engineering import extract_features
from src.triage.inference import run_triage_inference

async def main():
    settings = ServiceSettings()
    print("Loading ONNX model...")
    session = ort.InferenceSession("../../models/onnx/triage.onnx", providers=["CPUExecutionProvider"])
    
    print("Loading Encoder...")
    encoder = joblib.load("../../models/onnx/triage_encoder.pkl")
    
    alerts = [
        {
            "name": "Critical SSH Brute Force (Domain Controller)",
            "alert": {
                "alert_id": "test-1",
                "tenant_id": "default",
                "timestamp": "2026-03-30T15:00:00Z",
                "source": "wazuh",
                "normalized_title": "Brute force SSH login on domain controller",
                "severity": "Critical",
                "category": "authentication",
                "mitre_tactic": "Credential Access",
                "mitre_technique": "T1110",
                "suspicion_level": "High"
            },
            "asset_criticality": 0.9,
            "behavioral_score": 0.8
        },
        {
            "name": "Low-severity routine event",
            "alert": {
                "alert_id": "test-2",
                "tenant_id": "default",
                "timestamp": "2026-03-30T15:00:00Z",
                "source": "wazuh",
                "normalized_title": "Log rotation completed",
                "severity": "Low",
                "category": "system",
                "suspicion_level": "Low"
            },
            "asset_criticality": 0.2,
            "behavioral_score": 0.1
        },
        {
            "name": "SQL Injection Attempt",
            "alert": {
                "alert_id": "test-3",
                "tenant_id": "default",
                "timestamp": "2026-03-30T15:02:00Z",
                "source": "wazuh",
                "normalized_title": "SQL injection attempt detected on web application",
                "severity": "Critical",
                "category": "web",
                "mitre_tactic": "Initial Access",
                "mitre_technique": "T1190",
                "suspicion_level": "Critical"
            },
            "asset_criticality": 0.8,
            "behavioral_score": 0.9
        }
    ]
    
    for case in alerts:
        print(f"\n--- Testing: {case['name']} ---")
        feats = extract_features(case['alert'], encoder=encoder)
        result = await run_triage_inference(
            session=session,
            features=feats,
            alert_id=case['alert']['alert_id'],
            tenant_id=case['alert']['tenant_id'],
            asset_criticality=case['asset_criticality'],
            behavioral_score=case['behavioral_score'],
            settings=settings,
            suspicion_level=case['alert'].get('suspicion_level')
        )
        print(f"ML Score: {result.ml_score}")
        print(f"Risk Score: {result.risk_score}")
        print(f"Incident Grade: {result.incident_grade.value}")
        print(f"Auto Closed: {result.auto_closed}")

if __name__ == "__main__":
    asyncio.run(main())
