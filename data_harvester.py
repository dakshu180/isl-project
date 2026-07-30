import cv2
import mediapipe as mp
import os
import time

def calculate_dynamic_square(x_coords, y_coords, max_w, max_h):
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_w, box_h = x_max - x_min, y_max - y_min
    core_size = max(box_w, box_h)
    padded_size = int(core_size * 1.3)
    center_x = x_min + (box_w // 2)
    center_y = y_min + (box_h // 2)
    new_x_min = max(0, center_x - padded_size // 2)
    new_x_max = min(max_w, center_x + padded_size // 2)
    new_y_min = max(0, center_y - padded_size // 2)
    new_y_max = min(max_h, center_y + padded_size // 2)
    return new_x_min, new_y_min, new_x_max, new_y_max

def main():
    print("==================================================")
    print(" OMEGA-CLASS ACTIVE LEARNING HARVESTER")
    print("==================================================")
    
    # 1. Setup Dataset Directory
    base_dir = "custom_dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    target_word = input("Enter the word you want to teach the AI (e.g., ME, HELLO, YES): ").strip().upper()
    
    save_dir = os.path.join(base_dir, target_word)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"\n[SYSTEM] Target locked: [{target_word}]")
    print("[SYSTEM] Booting Camera. Press 'R' to start recording 300 frames. Press 'Q' to quit.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    is_recording = False
    frames_captured = 0
    TARGET_FRAMES = 300

    while True:
        ret, raw_frame = cap.read()
        if not ret: break
            
        h, w = raw_frame.shape[:2]
        frame = cv2.flip(raw_frame, 1)
        clean_frame = frame.copy() # Capture from here so UI lines aren't saved
        
        rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        
        # UI Elements
        cv2.putText(frame, f"LEARNING MODE: {target_word}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_coords = [int((1.0 - lm.x) * w) for lm in hand_landmarks.landmark]
                y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
                
                # Dynamic Square
                x_min, y_min, x_max, y_max = calculate_dynamic_square(x_coords, y_coords, w, h)
                
                # Draw Box for user
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 100), 2)
                
                if is_recording and x_max > x_min and y_max > y_min:
                    roi = clean_frame[y_min:y_max, x_min:x_max]
                    
                    if roi.size > 0:
                        # Save image to disk
                        file_path = os.path.join(save_dir, f"{target_word}_{int(time.time()*1000)}_{frames_captured}.jpg")
                        cv2.imwrite(file_path, roi)
                        frames_captured += 1
                        
                        # Draw Progress Bar
                        progress = int((frames_captured / TARGET_FRAMES) * 400)
                        cv2.rectangle(frame, (w//2 - 200, h - 60), (w//2 - 200 + progress, h - 40), (0, 255, 100), -1)
                        cv2.putText(frame, f"CAPTURING: {frames_captured}/{TARGET_FRAMES}", (w//2 - 100, h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
                        
                        if frames_captured >= TARGET_FRAMES:
                            is_recording = False
                            print(f"\n[SUCCESS] Successfully harvested {TARGET_FRAMES} images of '{target_word}'.")
                            print("Run the script again to add another word, or press Q to exit.")
        
        if not is_recording and frames_captured == 0:
             cv2.putText(frame, "Align hand in box, then press 'R' to record", (w//2 - 250, h - 40), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 150, 255), 1)
             
        cv2.imshow("Active Learning Harvester", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not is_recording and frames_captured < TARGET_FRAMES:
            is_recording = True
            print(f"[SYSTEM] Recording initiated for [{target_word}]...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()