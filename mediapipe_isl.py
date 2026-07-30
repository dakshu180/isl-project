import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import numpy as np

# 1. Redefine the exact Kaggle CNN Architecture
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
    # 2. System and Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Engine starting on: {device}")
    
    # Full A-Z and 0-9 Mapping
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 
               'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    
    model = ISLClassifier(num_classes=35).to(device)
    model.load_state_dict(torch.load('isl_cnn_model.pth', map_location=device))
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 3. Initialize MediaPipe Engine
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    # We restrict to 1 hand to keep inference lightning fast
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

    # 4. High-Performance I/O Stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("MediaPipe Engine active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # 5. Dynamic Bounding Box & Inference
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw the skeletal structure for UI feedback
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract geometric constraints (Adapted from provided source code)
                x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                
                # Apply 40px padding to capture the whole hand
                x_min, x_max = max(0, min(x_coords) - 40), min(w, max(x_coords) + 40)
                y_min, y_max = max(0, min(y_coords) - 40), min(h, max(y_coords) + 40)

                # Draw targeting box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 100), 2)

                # Ensure the box is valid before running inference
                if x_max > x_min and y_max > y_min:
                    # 6. Isolate the Hand (Crop)
                    roi = frame[y_min:y_max, x_min:x_max]
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PyTorch Tensor
                    pil_image = Image.fromarray(roi_rgb)
                    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                    
                    # 7. Run Neural Network Prediction
                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = F.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)
                        
                        predicted_class = classes[predicted_idx.item()]
                        confidence_score = confidence.item() * 100

                    # Render Telemetry 
                    if confidence_score > 50.0:
                        label = f"{predicted_class} ({confidence_score:.1f}%)"
                        cv2.putText(frame, label, (x_min, max(20, y_min - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow("MediaPipe ISL Core", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()