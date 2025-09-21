import requests
import json
from io import BytesIO
from PIL import Image

BASE = "https://c7erwbv7hsgrdy4yd6q7pblmh40thlde.lambda-url.eu-north-1.on.aws"

def create_test_image_bytes():
    """Create a simple test image and return as bytes"""
    img = Image.new('RGB', (224, 224), color=(255, 0, 0))
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

# Test basic endpoints with direct HTTP requests (not Lambda events)
print("Testing basic endpoints with direct HTTP...")

# Test GET requests
print("\n🔍 Testing GET requests:")
for endpoint in ["", "/health"]:
    try:
        resp = requests.get(f"{BASE}{endpoint}")
        print(f"GET {endpoint}: Status {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"GET {endpoint}: Error - {e}")

# Test POST requests  
print("\n🔍 Testing POST requests:")
for endpoint in ["", "/health"]:
    try:
        resp = requests.post(f"{BASE}{endpoint}")
        print(f"POST {endpoint}: Status {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"POST {endpoint}: Error - {e}")

# Test attack endpoint with multipart form data
print("\n🔍 Testing attack endpoint with multipart form:")
try:
    image_file = create_test_image_bytes()
    
    files = {'file': ('test.png', image_file, 'image/png')}
    data = {'epsilon': '0.1'}
    
    resp = requests.post(f"{BASE}/attack", files=files, data=data)
    print(f"Attack endpoint: Status {resp.status_code}")
    
    if resp.status_code == 200:
        response_data = resp.json()
        print("Success! Response:", json.dumps(response_data, indent=2))
    else:
        print(f"Error response: {resp.text}")
        
except Exception as e:
    print(f"Attack endpoint error: {e}")
    import traceback
    traceback.print_exc()