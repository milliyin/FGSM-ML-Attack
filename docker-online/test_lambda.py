import requests
import json
import base64
from io import BytesIO
from PIL import Image

BASE = "http://localhost:9000/2015-03-31/functions/function/invocations"

def make_event(path, method="GET", body=None, headers=None, is_base64=False):
    event_headers = {"content-type": "application/json"}
    if headers:
        event_headers.update(headers)
        
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "requestContext": {
            "http": {"method": method, "path": path, "sourceIp": "127.0.0.1"},
        },
        "headers": event_headers,
        "body": body,
        "isBase64Encoded": is_base64,
    }

def create_test_image_bytes():
    """Create a simple test image and return as bytes"""
    # Create a proper sized image (224x224 is common for ML models)
    img = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red image
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def create_multipart_binary():
    """Create multipart form data with proper binary handling"""
    boundary = b"----WebKitFormBoundary1234567890"
    
    # Get test image bytes
    image_bytes = create_test_image_bytes()
    
    # Build multipart form data as binary
    form_data = b""
    
    # File field
    form_data += b"--" + boundary + b"\r\n"
    form_data += b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
    form_data += b'Content-Type: image/png\r\n'
    form_data += b'\r\n'
    form_data += image_bytes  # Raw binary image data
    form_data += b'\r\n'
    
    # Epsilon field  
    form_data += b"--" + boundary + b"\r\n"
    form_data += b'Content-Disposition: form-data; name="epsilon"\r\n'
    form_data += b'\r\n'
    form_data += b'0.1'
    form_data += b'\r\n'
    
    # End boundary
    form_data += b"--" + boundary + b"--\r\n"
    
    return form_data, boundary.decode('ascii')

# Test basic endpoints
print("Testing basic endpoints...")

# Root
resp = requests.post(BASE, json=make_event("/"))
print("Root:", resp.json())

# Health
resp = requests.post(BASE, json=make_event("/health"))
print("Health:", resp.json())

# Attack endpoint with proper binary multipart
print("\nTesting attack endpoint with binary multipart form data...")

try:
    multipart_binary, boundary = create_multipart_binary()
    
    # Base64 encode the binary multipart data for Lambda
    multipart_body_b64 = base64.b64encode(multipart_binary).decode('ascii')
    
    # Create the Lambda event
    attack_event = make_event(
        "/attack",
        "POST",
        body=multipart_body_b64,
        headers={
            "content-type": f"multipart/form-data; boundary={boundary}",
            "content-length": str(len(multipart_binary))
        },
        is_base64=True
    )
    
    resp = requests.post(BASE, json=attack_event)
    response_data = resp.json()
    print("Attack response:", json.dumps(response_data, indent=2))
    
    # If successful, show key info
    if response_data.get("statusCode") == 200:
        body = json.loads(response_data.get("body", "{}"))
        if "original_prediction" in body:
            print(f"\n✅ Attack successful!")
            print(f"Original: {body['original_prediction']['class_name']} ({body['original_prediction']['confidence']:.2f})")
            print(f"Adversarial: {body['adversarial_prediction']['class_name']} ({body['adversarial_prediction']['confidence']:.2f})")
            print(f"Attack Success: {body['attack_success']}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback: Try creating a very simple test
    print("\n🔄 Trying fallback approach...")
    
    # Create the simplest possible valid PNG
    simple_img = Image.new('RGB', (32, 32), color='red')
    simple_bytes = BytesIO()
    simple_img.save(simple_bytes, format='PNG')
    simple_image_data = simple_bytes.getvalue()
    
    # Verify the image is valid
    try:
        test_img = Image.open(BytesIO(simple_image_data))
        print(f"✅ Created valid {test_img.format} image: {test_img.size} {test_img.mode}")
    except Exception as img_error:
        print(f"❌ Image creation failed: {img_error}")
        exit(1)
    
    # Create minimal multipart
    boundary = b"boundary123"
    minimal_form = (
        b"--boundary123\r\n"
        b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
        b'Content-Type: image/png\r\n'
        b'\r\n'
    ) + simple_image_data + (
        b'\r\n'
        b"--boundary123\r\n"
        b'Content-Disposition: form-data; name="epsilon"\r\n'
        b'\r\n'
        b'0.05\r\n'
        b"--boundary123--\r\n"
    )
    
    fallback_event = make_event(
        "/attack",
        "POST",
        body=base64.b64encode(minimal_form).decode('ascii'),
        headers={"content-type": "multipart/form-data; boundary=boundary123"},
        is_base64=True
    )
    
    resp = requests.post(BASE, json=fallback_event)
    print("Fallback response:", json.dumps(resp.json(), indent=2))