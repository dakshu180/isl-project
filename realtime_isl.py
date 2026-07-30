import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# 1. Redefine the exact same architecture we built in Kaggle
class ISLClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ISLClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def main():
    # 2. Setup Device and Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}")
    
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 
               'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    model = ISLClassifier(num_classes=35).to(device)
    
    # Load the weights (ensure the .pth file is in the same directory)
    try:
        model.load_state_dict(torch.load('isl_cnn_model.pth', map_location=device))
        model.eval() # Set to evaluation mode
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Error: 'isl_cnn_model.pth' not found. Please put it in the same folder as this script.")
        return

    # 3. Define the preprocessing pipeline
    # This must perfectly match what we did in Kaggle
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 4. Initialize Webcam
    cap = cv2.VideoCapture(0) # 0 is usually the default laptop webcam
    
    print("Starting webcam... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally like a mirror for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Define the Region of Interest (ROI) coordinates (x1, y1, x2, y2)
        # We draw a 300x300 pixel box on the right side of the screen
        roi_x1, roi_y1 = 350, 100
        roi_x2, roi_y2 = 650, 400
        
        # Draw the bounding box on the main frame
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        
        # Crop the ROI from the frame
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # OpenCV uses BGR format, PyTorch (and our PIL pipeline) expects RGB
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image, preprocess, and add batch dimension
        pil_image = Image.fromarray(roi_rgb)
        input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
        
        # 5. Run Inference
        with torch.no_grad():
            output = model(input_tensor)
            # Apply softmax to get confidence scores
            probabilities = F.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = classes[predicted_idx.item()]
            confidence_score = confidence.item() * 100

        # 6. Display the Results
        # Only show the prediction if the model is somewhat confident
        if confidence_score > 60.0:
            label_text = f"Sign: {predicted_class} ({confidence_score:.1f}%)"
            cv2.putText(frame, label_text, (roi_x1, roi_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("ISL Real-Time Translator", frame)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()