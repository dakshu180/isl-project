import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque, Counter

# ==============================================================================
# 1. CORE CNN ARCHITECTURE
# ==============================================================================
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

# ==============================================================================
# 2. OMEGA-CLASS UI RENDERER 
# ==============================================================================
def draw_cyber_targeting_box(frame, x_min, y_min, x_max, y_max, color):
    """Renders a high-tech tracking reticle with animated scanning elements."""
    thickness = 2
    length = 20
    
    # Structural Corners
    cv2.line(frame, (x_min, y_min), (x_min + length, y_min), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_min), (x_min, y_min + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_min), (x_max - length, y_min), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_min), (x_max, y_min + length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_max), (x_min + length, y_max), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_max), (x_min, y_max - length), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_max), (x_max - length, y_max), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_max), (x_max, y_max - length), color, thickness, cv2.LINE_AA)
    
    # Animated Laser Scan Line
    box_height = y_max - y_min
    if box_height > 0:
        scan_y_offset = int((math.sin(time.time() * 5) + 1) / 2 * box_height)
        scan_y = y_min + scan_y_offset
        cv2.line(frame, (x_min + 5, scan_y), (x_max - 5, scan_y), color, 1)

def draw_interface_panels(frame, w, h):
    """Draws sleek transparent glass-morphism panels for data display."""
    overlay = frame.copy()
    # Top Telemetry Header
    cv2.rectangle(overlay, (0, 0), (w, 55), (10, 10, 15), -1)
    cv2.line(overlay, (0, 55), (w, 55), (0, 255, 150), 1) 
    
    # Bottom Sentence Console
    cv2.rectangle(overlay, (0, h - 85), (w, h), (10, 10, 15), -1)
    cv2.line(overlay, (0, h - 85), (w, h - 85), (0, 255, 150), 1)
    
    # Blend overlay with original frame (80% opacity)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

# ==============================================================================
# 3. MAIN ENGINE LOOP
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Central Interface Boot Sequence Initialized on {device}.")
    
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

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2, 
        min_detection_confidence=0.75, 
        min_tracking_confidence=0.75
    )

    HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                  (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), 
                  (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    prev_time = time.time()
    font = cv2.FONT_HERSHEY_DUPLEX

    # State Variables
    invert_hands = True # Hotkey 'H' toggles this
    memory_span = 7
    prediction_history = {"Left": deque(maxlen=memory_span), "Right": deque(maxlen=memory_span)}
    
    current_word = ""
    current_sentence = ""
    last_confirmed_char = None
    frames_held = 0
    CONFIRM_THRESH = 15 
    no_hand_frames = 0
    SPACE_THRESH = 25 

    COLOR_CYAN = (255, 255, 0)
    COLOR_MAGENTA = (200, 50, 255)

    while True:
        ret, raw_frame = cap.read()
        if not ret: break
            
        h, w = raw_frame.shape[:2] 
        frame = cv2.flip(raw_frame, 1)
        
        # [PRECISION FIX] We keep a clean copy of the frame BEFORE drawing UI elements on it.
        # This prevents the CNN from looking at drawn lines, text, or HUD boxes.
        inference_frame = frame.copy() 

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # 1. Draw UI Backgrounds
        draw_interface_panels(frame, w, h)

        # 2. Process Vision Core
        rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        active_hands = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        
        # Sentence Auto-Spacing
        if active_hands == 0:
            no_hand_frames += 1
            if no_hand_frames > SPACE_THRESH and len(current_word) > 0:
                current_sentence += current_word + " "
                current_word = ""
                last_confirmed_char = None
        else:
            no_hand_frames = 0

        # Top Telemetry Rendering
        cv2.putText(frame, "AUTONOMOUS TRACKING CORE", (20, 35), font, 0.6, COLOR_CYAN, 1, cv2.LINE_AA)
        
        center_text = f"SUBJECTS ACQUIRED: {active_hands}"
        tw = cv2.getTextSize(center_text, font, 0.6, 1)[0][0]
        cv2.putText(frame, center_text, ((w - tw) // 2, 35), font, 0.6, COLOR_MAGENTA if active_hands > 0 else (100,100,100), 1, cv2.LINE_AA)
        
        fps_text = f"FPS: {fps:.1f}"
        fw = cv2.getTextSize(fps_text, font, 0.6, 1)[0][0]
        cv2.putText(frame, fps_text, (w - fw - 20, 35), font, 0.6, COLOR_CYAN, 1, cv2.LINE_AA)

        if result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # 3. Dynamic Hand Inversion Logic
                raw_hand = result.multi_handedness[idx].classification[0].label
                if invert_hands:
                    actual_hand = "Right" if raw_hand == "Left" else "Left"
                else:
                    actual_hand = raw_hand
                
                theme_color = COLOR_CYAN if actual_hand == "Left" else COLOR_MAGENTA

                points = []
                x_coords, y_coords = [], []
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    points.append((cx, cy))
                    x_coords.append(cx)
                    y_coords.append(cy)

                # Render Holographic Skeleton
                for p1, p2 in HAND_BONES:
                    cv2.line(frame, points[p1], points[p2], (255, 255, 255), 1, cv2.LINE_AA)
                for (cx, cy) in points:
                    cv2.circle(frame, (cx, cy), 3, theme_color, cv2.FILLED)

                # Raw Bounding Box Calculation
                padding = 40
                x_min = max(0, min(x_coords) - padding)
                x_max = min(w, max(x_coords) + padding)
                y_min = max(0, min(y_coords) - padding)
                y_max = min(h, max(y_coords) + padding)
                
                # Render Cybernetic Box (Only on visual frame)
                draw_cyber_targeting_box(frame, x_min, max(55, y_min), x_max, min(h-85, y_max), theme_color)
                
                # Render XYZ coordinate telemetry for a highly technical aesthetic
                wrist_x, wrist_y, wrist_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                cv2.putText(frame, f"XYZ: {wrist_x:.2f}, {wrist_y:.2f}, {wrist_z:.2f}", 
                            (x_min, min(h-90, y_max + 15)), font, 0.4, theme_color, 1, cv2.LINE_AA)

                # 4. [PRECISION FIX] Perfect Aspect Ratio Padding
                if x_max > x_min and y_max > y_min:
                    # Crop from the CLEAN frame
                    roi = inference_frame[y_min:y_max, x_min:x_max] 
                    if roi.size == 0: continue
                    
                    # Pad the ROI to be a perfect square, preventing PyTorch from squishing the image
                    roi_h, roi_w = roi.shape[:2]
                    target_size = max(roi_h, roi_w)
                    pad_t = (target_size - roi_h) // 2
                    pad_b = target_size - roi_h - pad_t
                    pad_l = (target_size - roi_w) // 2
                    pad_r = target_size - roi_w - pad_l
                    
                    square_roi = cv2.copyMakeBorder(roi, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0,0,0))
                    
                    pil_image = Image.fromarray(cv2.cvtColor(square_roi, cv2.COLOR_BGR2RGB))
                    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(input_tensor)
                        confidence, predicted_idx = torch.max(F.softmax(output, dim=1), 1)
                        predicted_class = classes[predicted_idx.item()]
                        score = confidence.item() * 100

                    if score > 55.0:
                        prediction_history[actual_hand].append(predicted_class)
                    else:
                        prediction_history[actual_hand].append(None) 

                    if len(prediction_history[actual_hand]) > 0:
                        most_common_pred = Counter(prediction_history[actual_hand]).most_common(1)[0][0]
                        
                        # Sentence Typing Logic (Restricted to Hand 1)
                        if idx == 0:
                            if most_common_pred is not None:
                                if prediction_history[actual_hand].count(most_common_pred) >= memory_span - 2:
                                    frames_held += 1
                                    if frames_held == CONFIRM_THRESH:
                                        if most_common_pred != last_confirmed_char:
                                            current_word += most_common_pred
                                            last_confirmed_char = most_common_pred
                                else:
                                    frames_held = max(0, frames_held - 1) 
                            else:
                                last_confirmed_char = None
                                frames_held = 0

                        # Draw Dynamic Target Labels
                        if most_common_pred is not None:
                            label = f"{actual_hand} [{most_common_pred}] {score:.0f}%"
                            lw = cv2.getTextSize(label, font, 0.5, 1)[0][0]
                            
                            plate_y = max(80, y_min - 15)
                            
                            # Loading Ring UI
                            if idx == 0 and frames_held > 0 and frames_held <= CONFIRM_THRESH:
                                progress = int(360 * (frames_held / CONFIRM_THRESH))
                                cv2.ellipse(frame, (x_min + lw + 25, plate_y - 5), (6, 6), -90, 0, progress, theme_color, 2, cv2.LINE_AA)

                            cv2.rectangle(frame, (x_min, plate_y - 15), (x_min + lw + 10, plate_y + 5), (15, 15, 15), -1)
                            cv2.rectangle(frame, (x_min, plate_y - 15), (x_min + lw + 10, plate_y + 5), theme_color, 1)
                            cv2.putText(frame, label, (x_min + 5, plate_y), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

        # 5. Render Command Console
        cursor = "_" if int(time.time() * 2) % 2 == 0 else " "
        cv2.putText(frame, "CONSOLE //", (20, h - 45), font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{current_sentence}{current_word}{cursor}", (130, h - 43), font, 0.8, COLOR_CYAN, 2, cv2.LINE_AA)
        
        cv2.putText(frame, "[H] Calibrate L/R  |  [BACKSPACE] Erase  |  [C] Wipe", (w - 480, h - 30), font, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Multi-Hand ISL Intelligence Core", frame)
        
        # Keyboard Hardware Intercepts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('h'): # HOTKEY TO FIX LEFT/RIGHT PARADOX
            invert_hands = not invert_hands
            print(f"[SYSTEM] Hardware Calibration Switched. Invert State: {invert_hands}")
        elif key == ord('c'): 
            current_sentence = ""
            current_word = ""
            last_confirmed_char = None
        elif key == 8 or key == 127: 
            if len(current_word) > 0:
                current_word = current_word[:-1]
                last_confirmed_char = None
            elif len(current_sentence) > 0:
                current_sentence = current_sentence[:-1]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()