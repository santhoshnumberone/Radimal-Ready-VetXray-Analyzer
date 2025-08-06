import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import xgboost as xgb
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# --- File Paths ---
CNN_MODEL_PATH = 'models/cnn_model.pth'
XGB_MODEL_PATH = 'models/structured_model.json'
METADATA_CSV_PATH = 'data/sample_labels.csv'

# --- Load metadata once ---
print("Loading metadata...")
metadata_df = pd.read_csv(METADATA_CSV_PATH)

# Clean and preprocess the age data (remove 'Y' suffix)
metadata_df['Patient Age'] = metadata_df['Patient Age'].str.replace('Y', '', regex=False).astype(int)

# Encode gender using LabelEncoder
le = LabelEncoder()
metadata_df['Patient Gender'] = le.fit_transform(metadata_df['Patient Gender'])

# --- Image preprocessing transforms ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Load models once ---
print("Loading CNN model...")
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

# Load CNN model (MobileNetV2 with custom classifier)
cnn_model = models.mobilenet_v2(weights=None)
cnn_model.classifier[1] = nn.Linear(cnn_model.classifier[1].in_features, 1)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.to(device)
cnn_model.eval()

print("Loading XGBoost model...")
# Load XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_MODEL_PATH)

print("Models loaded successfully!")

def predict_xray(image_path, cnn_weight=0.7, xgb_weight=0.3):
    """
    Perform hybrid prediction on an X-ray image.
    
    Args:
        image_path (str): Path to the X-ray image file
        cnn_weight (float): Weight for CNN prediction (default: 0.7)
        xgb_weight (float): Weight for XGBoost prediction (default: 0.3)
    
    Returns:
        tuple: (final_prediction, cnn_score, xgb_score)
            - final_prediction: Final weighted probability score
            - cnn_score: CNN probability score
            - xgb_score: XGBoost probability score
    """
    
    # Get the image filename to look up metadata
    image_filename = os.path.basename(image_path)
    
    # Retrieve structured data from metadata CSV
    try:
        patient_row = metadata_df[metadata_df['Image Index'] == image_filename].iloc[0]
        patient_age = patient_row['Patient Age']
        patient_gender = patient_row['Patient Gender']  # Already encoded
    except IndexError:
        raise ValueError(f"Image {image_filename} not found in metadata CSV")
    
    # --- CNN Prediction ---
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get CNN prediction
        with torch.no_grad():
            cnn_output = cnn_model(image_tensor)
            cnn_score = torch.sigmoid(cnn_output).item()
            
    except Exception as e:
        raise RuntimeError(f"Error processing image: {str(e)}")
    
    # --- XGBoost Prediction ---
    try:
        # Prepare structured data features
        structured_features = [[patient_age, patient_gender]]
        
        # Get XGBoost prediction probability for positive class
        xgb_score = xgb_model.predict_proba(structured_features)[0][1]
        
    except Exception as e:
        raise RuntimeError(f"Error with XGBoost prediction: {str(e)}")
    
    # --- Combine predictions using weighted average ---
    final_prediction = (cnn_score * cnn_weight) + (xgb_score * xgb_weight)
    
    return final_prediction, cnn_score, xgb_score


# --- Example Usage ---
if __name__ == "__main__":
    # Test the prediction function
    sample_image_path = 'data/images/00000013_005.png'
    
    try:
        print(f"\nTesting prediction for: {sample_image_path}")
        
        # Get prediction
        final_prob, cnn_prob, xgb_prob = predict_xray(sample_image_path)
        
        # Display results
        print("\n" + "="*40)
        print("PREDICTION RESULTS")
        print("="*40)
        print(f"CNN Score (Image): {cnn_prob:.4f}")
        print(f"XGBoost Score (Metadata): {xgb_prob:.4f}")
        print(f"Final Weighted Score: {final_prob:.4f}")
        print("-"*40)
        
        # Convert to binary prediction
        prediction_label = "Abnormal" if final_prob > 0.5 else "Normal"
        confidence = abs(final_prob - 0.5) * 200  # Convert to percentage
        
        print(f"Final Prediction: {prediction_label}")
        print(f"Confidence: {confidence:.1f}%")
        print("="*40)
        
    except FileNotFoundError:
        print(f"Error: Image file not found at {sample_image_path}")
        print("Please update the path to point to a valid image file.")
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
