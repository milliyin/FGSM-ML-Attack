import os
import json
import base64
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Import your FGSM attack class
from fgsm import Attack

app = FastAPI(title="FGSM Attack API", description="FastAPI service for FGSM adversarial attacks on MNIST")

handler = Mangum(app)

# # CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Lambda
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model at module level for Lambda reuse
device = torch.device("cpu")  # Lambda typically uses CPU
model = None

def initialize_model():
    global model
    if model is None:
        model = SimpleNet().to(device)
        
        # Try to load pre-trained model
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "mnist_model.pt")
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                print("Loaded pre-trained model")
            except Exception as e:
                print(f"Error loading model: {e}")
                # For Lambda, you should pre-train and include the model file
                raise HTTPException(status_code=500, detail="Model file not found or corrupted")
        else:
            raise HTTPException(status_code=500, detail="Pre-trained model file not found")
        
        model.eval()
    return model

# Transform for preprocessing
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

CLASS_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor.to(device)

def tensor_to_base64(tensor: torch.Tensor) -> str:
    tensor = tensor.squeeze(0).cpu()
    tensor = torch.clamp(tensor, 0, 1)
    array = tensor.squeeze(0).numpy()
    array = (array * 255).astype(np.uint8)
    
    image = Image.fromarray(array, mode='L')
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

def get_prediction(model_output: torch.Tensor) -> dict:
    probabilities = F.softmax(model_output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    
    return {
        "class": CLASS_LABELS[predicted_class],
        "class_index": predicted_class,
        "confidence": float(confidence)
    }

@app.post("/attack")
async def fgsm_attack_endpoint(
    file: UploadFile = File(...),
    epsilon: Optional[float] = Form(0.1)
):
    # Initialize model on first request
    model = initialize_model()
    
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload PNG or JPEG image.")
    
    if epsilon < 0 or epsilon > 1:
        raise HTTPException(status_code=400, detail="Epsilon must be between 0 and 1")
    
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes).clone().detach().to(device)
        image_tensor.requires_grad_(True)

        with torch.no_grad():
            original_output = model(image_tensor)
            original_prediction = get_prediction(original_output)

        target = torch.tensor(
            [original_prediction["class_index"]],
            dtype=torch.long,
            device=device
        )

        attack = Attack(model, epsilon=epsilon)
        criterion = nn.CrossEntropyLoss()
        
        adversarial_tensor = attack.generate_adversarial(
            image_tensor.clone().detach().to(device).requires_grad_(True), 
            target, 
            criterion
        )
        
        with torch.no_grad():
            adversarial_output = model(adversarial_tensor)
            adversarial_prediction = get_prediction(adversarial_output)
        
        adversarial_base64 = tensor_to_base64(adversarial_tensor.detach())
        attack_success = original_prediction["class_index"] != adversarial_prediction["class_index"]
        
        response = {
            "original_prediction": original_prediction,
            "adversarial_prediction": adversarial_prediction,
            "adversarial_image_base64": adversarial_base64,
            "attack_success": attack_success
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "FGSM Attack API",
        "endpoints": {"/attack": "POST - Perform FGSM attack"}
    }

@app.get("/health")
async def health_check():
    model = initialize_model()
    return {"status": "healthy", "device": str(device), "model_loaded": model is not None}

def lambda_handler(event, context):
    """
    AWS Lambda handler that forwards requests to FastAPI via Mangum
    """
    return handler(event, context)