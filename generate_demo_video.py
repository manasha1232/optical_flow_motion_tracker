import os
import cv2
import numpy as np

def generate_synthetic_motion_video(output_path="input/sample_motion_video.mp4", num_frames=150, width=800, height=600, fps=30):
    """
    Generates a synthetic video with multiple moving objects (circles, rectangles, textures)
    to demonstrate and benchmark Lucas-Kanade and Farneback optical flow tracking.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try MP4V codec, fallback to MJPG if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    print(f"[+] Generating synthetic motion video: '{output_path}' ({num_frames} frames)...")
    
    # Object 1: Bouncing Ball (moves diagonally)
    x1, y1 = 100, 100
    vx1, vy1 = 6, 4
    r1 = 35
    
    # Object 2: Horizontal Moving Car (moves left to right)
    x2, y2 = 50, 450
    vx2 = 8
    
    # Object 3: Circular Orbiting Satellite
    center_x, center_y = 550, 250
    orbit_r = 90
    
    # Static background texture (grid + text)
    bg_static = np.ones((height, width, 3), dtype=np.uint8) * 35
    # Grid lines
    for x in range(0, width, 50):
        cv2.line(bg_static, (x, 0), (x, height), (50, 50, 50), 1)
    for y in range(0, height, 50):
        cv2.line(bg_static, (0, y), (width, y), (50, 50, 50), 1)
        
    cv2.putText(bg_static, "OPTICAL FLOW TEST BENCHMARK", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                
    for frame_idx in range(num_frames):
        frame = bg_static.copy()
        
        # --- Update Object 1 (Bouncing Ball) ---
        x1 += vx1
        y1 += vy1
        if x1 - r1 <= 50 or x1 + r1 >= width - 50:
            vx1 *= -1
        if y1 - r1 <= 50 or y1 + r1 >= height - 50:
            vy1 *= -1
            
        cv2.circle(frame, (int(x1), int(y1)), r1, (0, 215, 255), -1) # Yellow ball
        cv2.circle(frame, (int(x1), int(y1)), r1, (255, 255, 255), 2)
        # Inner pattern for Shi-Tomasi feature tracking
        cv2.circle(frame, (int(x1), int(y1)), 10, (20, 20, 20), -1)
        cv2.line(frame, (int(x1) - 20, int(y1)), (int(x1) + 20, int(y1)), (255, 255, 255), 2)
        cv2.line(frame, (int(x1), int(y1) - 20), (int(x1), int(y1) + 20), (255, 255, 255), 2)
        
        # --- Update Object 2 (Horizontal Moving Car) ---
        x2 = (x2 + vx2) % (width + 100)
        car_x = int(x2) - 50
        cv2.rectangle(frame, (car_x, y2), (car_x + 90, y2 + 40), (0, 120, 255), -1) # Orange car body
        cv2.rectangle(frame, (car_x + 15, y2 - 20), (car_x + 65, y2), (0, 80, 200), -1) # Cabin
        cv2.circle(frame, (car_x + 20, y2 + 40), 12, (200, 200, 200), -1) # Wheels
        cv2.circle(frame, (car_x + 70, y2 + 40), 12, (200, 200, 200), -1)
        
        # --- Update Object 3 (Orbiting Satellite) ---
        angle = (frame_idx / 15.0)
        sat_x = int(center_x + orbit_r * np.cos(angle))
        sat_y = int(center_y + orbit_r * np.sin(angle))
        
        # Draw orbit ring line
        cv2.circle(frame, (center_x, center_y), orbit_r, (70, 70, 70), 1, lineType=cv2.LINE_AA)
        cv2.rectangle(frame, (sat_x - 18, sat_y - 18), (sat_x + 18, sat_y + 18), (255, 100, 0), -1) # Blue satellite
        cv2.circle(frame, (sat_x, sat_y), 6, (255, 255, 255), -1)
        
        # Frame Number Stamp
        cv2.putText(frame, f"FRAME: {frame_idx:03d} / {num_frames}", (width - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                    
        out.write(frame)
        
    out.release()
    print(f"[OK] Synthetic video saved to '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_synthetic_motion_video()
