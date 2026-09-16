import os
import sys
from dotenv import load_dotenv

# 🌟 Direct path mapping setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest
from fastapi.testclient import TestClient

def test_expose_startup_error():
    """🌟 DIAGNOSTIC TEST: Forces pytest to print out the real startup error."""
    print("\n--- DEBUGGING STARTUP ERROR ---")
    try:
        from app.main import app
        # This will trigger the actual exception out in the open
        with TestClient(app) as client:
            response = client.post("/ask", json={"question": "ping"})
            print(f"API status code: {response.status_code}")
            print(f"API body response: {response.text}")
    except Exception as e:
        import traceback
        print("\n❌ CRITICAL CRASH DETECTED ON STARTUP:")
        traceback.print_exc()
        print("--------------------------------\n")
        raise e
