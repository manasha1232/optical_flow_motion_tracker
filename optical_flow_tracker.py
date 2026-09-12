#!/usr/bin/env python3
"""
===============================================================================
Optical Flow Motion Vector Tracker
Day 09 - 30-Day Computer Vision & Deep Learning Challenge
===============================================================================
Author: Computer Vision & AI Agent
Technologies: OpenCV, NumPy, Lucas-Kanade (LK), Farneback Dense Flow, Motion Vectors

Description:
    Real-time Motion Vector Tracker implementing both Sparse Lucas-Kanade (LK) 
    Pyramidal Optical Flow with feature re-seeding and Dense Farneback Optical 
    Flow with HSV direction heatmaps and motion vector quiver overlay.
===============================================================================
"""

import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np


class LucasKanadeTracker:
    """
    Sparse Optical Flow Tracker using Shi-Tomasi corner detection 
    and Lucas-Kanade pyramidal tracking with forward-backward error filtering.
    """
    def __init__(self, max_corners=200, quality_level=0.3, min_distance=7, block_size=7, trail_length=20):
        self.max_corners = max_corners
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.trail_length = trail_length
        self.tracks = [] # List of point trajectory histories
        self.prev_gray = None
        self.colors = np.random.randint(0, 255, (max_corners, 3), dtype=np.uint8)
        
    def detect_features(self, gray, mask=None):
        """Detects new Shi-Tomasi feature corners."""
        pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if pts is not None:
            return pts.reshape(-1, 2)
        return np.empty((0, 2), dtype=np.float32)
        
    def update(self, frame_bgr):
        """
        Updates optical flow tracking for the current frame.
        
        Returns:
            tuple: (vis_frame, frame_stats)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        vis = frame_bgr.copy()
        
        num_active_points = 0
        avg_speed = 0.0
        max_speed = 0.0
        directions = {"LEFT": 0, "RIGHT": 0, "UP": 0, "DOWN": 0, "STATIONARY": 0}
        
        if self.prev_gray is not None and len(self.tracks) > 0:
            # Prepare previous points
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            
            # Forward Lucas-Kanade flow: p0 -> p1
            p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **self.lk_params)
            
            # Backward Lucas-Kanade flow for validation: p1 -> p0_back
            p0_back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **self.lk_params)
            
            # Forward-backward error calculation: ||p0 - p0_back||
            d = abs(p0 - p0_back).reshape(-1, 2).max(axis=1)
            good_pts = (d < 1.0) & (st.reshape(-1) == 1)
            
            new_tracks = []
            speeds = []
            
            for i, (tr, (x, y), good) in enumerate(zip(self.tracks, p1.reshape(-1, 2), good_pts)):
                if not good:
                    continue
                    
                prev_x, prev_y = tr[-1]
                dx = x - prev_x
                dy = y - prev_y
                dist = float(np.hypot(dx, dy))
                speeds.append(dist)
                
                # Classify direction
                if dist < 0.5:
                    directions["STATIONARY"] += 1
                elif abs(dx) > abs(dy):
                    directions["RIGHT" if dx > 0 else "LEFT"] += 1
                else:
                    directions["DOWN" if dy > 0 else "UP"] += 1
                    
                tr.append((float(x), float(y)))
                if len(tr) > self.trail_length:
                    del tr[0]
                new_tracks.append(tr)
                
                # Draw motion trajectory trail
                color = self.colors[i % len(self.colors)].tolist()
                pts_arr = np.int32(tr).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts_arr], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(vis, (int(x), int(y)), 4, color, -1, lineType=cv2.LINE_AA)
                
            self.tracks = new_tracks
            num_active_points = len(self.tracks)
            if len(speeds) > 0:
                avg_speed = float(np.mean(speeds))
                max_speed = float(np.max(speeds))
                
        # Feature Re-seeding: Re-detect feature points if point count drops low
        if len(self.tracks) < self.max_corners // 2:
            mask = np.ones_like(gray) * 255
            for tr in self.tracks:
                x, y = int(tr[-1][0]), int(tr[-1][1])
                cv2.circle(mask, (x, y), 7, 0, -1)
                
            new_pts = self.detect_features(gray, mask=mask)
            for pt in new_pts:
                self.tracks.append([(float(pt[0]), float(pt[1]))])
                
        self.prev_gray = gray
        
        dominant_dir = max(directions, key=directions.get) if num_active_points > 0 else "STATIONARY"
        
        stats = {
            "num_active_points": num_active_points,
            "avg_speed_px": round(avg_speed, 2),
            "max_speed_px": round(max_speed, 2),
            "dominant_direction": dominant_dir,
            "direction_counts": directions
        }
        
        return vis, stats


class FarnebackDenseTracker:
    """
    Dense Optical Flow Tracker computing frame-wide velocity vector fields, 
    HSV color motion maps, and vector quiver grid overlays.
    """
    def __init__(self, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0
        )
        self.prev_gray = None
        
    def update(self, frame_bgr, step=16):
        """
        Computes dense optical flow field for frame.
        
        Args:
            frame_bgr (np.ndarray): Input BGR frame.
            step (int): Grid sampling step for quiver vector arrows.
            
        Returns:
            tuple: (hsv_heatmap_bgr, quiver_overlay_bgr, frame_stats)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        if self.prev_gray is None:
            self.prev_gray = gray
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            return frame_bgr.copy(), frame_bgr.copy(), {
                "avg_speed_px": 0.0, "max_speed_px": 0.0, "dominant_direction": "STATIONARY"
            }
            
        # Compute Farneback Optical Flow (flow field of shape HxWx2)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, **self.params)
        fx, fy = flow[..., 0], flow[..., 1]
        
        # Convert Cartesian vectors to Polar coordinates (magnitude and angle in degrees)
        mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        
        # 1. Build HSV Motion Heatmap
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang / 2).astype(np.uint8) # Hue represents angle [0..180]
        hsv[..., 1] = 255                        # Full Saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # Value represents magnitude
        hsv_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 2. Build Quiver Arrow Vector Grid Overlay
        quiver_vis = frame_bgr.copy()
        y_grid, x_grid = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx_sub = fx[y_grid, x_grid]
        fy_sub = fy[y_grid, x_grid]
        mag_sub = mag[y_grid, x_grid]
        
        lines = np.vstack([x_grid, y_grid, x_grid + fx_sub, y_grid + fy_sub]).T.reshape(-1, 2, 2)
        lines = np.int32(lines)
        
        for (x1, y1), (x2, y2), m in zip(lines[:, 0], lines[:, 1], mag_sub):
            if m > 1.0: # Filter out subtle noise vectors
                cv2.arrowedLine(quiver_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1, tipLength=0.3)
                cv2.circle(quiver_vis, (int(x1), int(y1)), 1, (0, 0, 255), -1)
                
        # Calculate statistics
        avg_speed = float(np.mean(mag))
        max_speed = float(np.max(mag))
        
        # Direction classification via angle histogram
        directions = {"RIGHT": 0, "UP": 0, "LEFT": 0, "DOWN": 0}
        moving_mask = mag > 1.5
        if np.any(moving_mask):
            moving_angles = ang[moving_mask]
            directions["RIGHT"] = int(np.sum((moving_angles >= 315) | (moving_angles < 45)))
            directions["UP"]    = int(np.sum((moving_angles >= 45)  & (moving_angles < 135)))
            directions["LEFT"]  = int(np.sum((moving_angles >= 135) & (moving_angles < 225)))
            directions["DOWN"]  = int(np.sum((moving_angles >= 225) & (moving_angles < 315)))
            dominant_dir = max(directions, key=directions.get)
        else:
            dominant_dir = "STATIONARY"
            
        self.prev_gray = gray
        
        stats = {
            "avg_speed_px": round(avg_speed, 2),
            "max_speed_px": round(max_speed, 2),
            "dominant_direction": dominant_dir
        }
        
        return hsv_bgr, quiver_vis, stats


def render_hud_dashboard(lk_vis, dense_hsv, quiver_vis, stats, fps=0.0):
    """
    Renders a 3-panel analytics dashboard montage containing:
    [Panel 1: Lucas-Kanade Feature Trails] | [Panel 2: Dense Farneback HSV Map] | [Panel 3: Vector Quiver Grid + HUD Telemetry]
    
    Args:
        lk_vis (np.ndarray): Lucas-Kanade annotated frame.
        dense_hsv (np.ndarray): Dense Optical Flow HSV heatmap.
        quiver_vis (np.ndarray): Quiver vector grid frame.
        stats (dict): Current frame telemetry statistics.
        fps (float): Current frame rate.
        
    Returns:
        np.ndarray: Combined 3-panel montage frame.
    """
    target_h = 400
    
    def resize_h(img, h):
        aspect = img.shape[1] / float(img.shape[0])
        return cv2.resize(img, (int(h * aspect), h), interpolation=cv2.INTER_AREA)
        
    p1 = resize_h(lk_vis, target_h)
    p2 = resize_h(dense_hsv, target_h)
    p3 = resize_h(quiver_vis, target_h)
    
    def add_header(img, title, subtitle="", color=(40, 40, 40)):
        h, w = img.shape[:2]
        hdr = np.zeros((45, w, 3), dtype=np.uint8)
        hdr[:] = color
        cv2.putText(hdr, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        if subtitle:
            cv2.putText(hdr, subtitle, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, lineType=cv2.LINE_AA)
        return np.vstack([hdr, img])
        
    p1_hdr = add_header(p1, "[1] LUCAS-KANADE SPARSE TRAILS", "Shi-Tomasi Corners & Flow Trails", (40, 60, 140))
    p2_hdr = add_header(p2, "[2] FARNEBACK DENSE HSV MAP", "Hue=Direction, Val=Magnitude", (140, 60, 40))
    p3_hdr = add_header(p3, "[3] QUIVER VECTOR GRID & HUD", "Flow Velocity Field Overlays", (40, 120, 60))
    
    divider = np.zeros((p1_hdr.shape[0], 5, 3), dtype=np.uint8)
    divider[:] = (180, 180, 180)
    
    montage = np.hstack([p1_hdr, divider, p2_hdr, divider, p3_hdr])
    
    # Overlay Global Telemetry Banner on bottom of montage
    banner_h = 50
    banner = np.zeros((banner_h, montage.shape[1], 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    
    active_pts = stats.get("num_active_points", "N/A")
    avg_sp = stats.get("avg_speed_px", 0.0)
    max_sp = stats.get("max_speed_px", 0.0)
    dom_dir = stats.get("dominant_direction", "STATIONARY")
    
    info_str = f"FPS: {fps:.1f} | Active Vectors: {active_pts} | Avg Speed: {avg_sp} px/f | Peak Speed: {max_sp} px/f | Dominant Motion: {dom_dir}"
    cv2.putText(banner, info_str, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, lineType=cv2.LINE_AA)
    
    final_montage = np.vstack([montage, banner])
    return final_montage


def process_video_stream(video_source, output_dir="output", algorithm="dual", max_frames=300):
    """
    Processes input video stream, tracking optical flow and saving output videos and metadata.
    
    Args:
        video_source (str or int): Path to video file or camera index (0).
        output_dir (str): Directory to save outputs.
        algorithm (str): 'lk', 'dense', or 'dual'.
        max_frames (int): Maximum frames to process for offline videos.
        
    Returns:
        dict: Processed telemetry summary stats.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    is_live = str(video_source).lower() in ["camera", "webcam", "0"]
    cap_src = 0 if is_live else video_source
    
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_source}")
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
        
    base_name = "live_camera" if is_live else os.path.splitext(os.path.basename(video_source))[0]
    out_video_path = os.path.join(output_dir, f"{base_name}_optical_flow_{algorithm}.mp4")
    
    # Estimate montage width for video writer
    montage_w = int((400 * (width / float(height))) * 3) + 10
    montage_h = 400 + 45 + 50 # panel height + header + bottom banner
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (montage_w, montage_h))
    
    if not writer.isOpened():
        out_video_path = out_video_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (montage_w, montage_h))
        
    lk_tracker = LucasKanadeTracker(max_corners=200, trail_length=25)
    dense_tracker = FarnebackDenseTracker()
    
    telemetry_logs = []
    frame_count = 0
    start_time = time.time()
    
    print(f"\n[+] Processing Video Optical Flow Stream: '{base_name}' ({width}x{height} @ {fps_in:.1f} FPS)")
    print(f"  - Mode: {algorithm.upper()} | Output: '{out_video_path}'")
    
    prev_frame_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        curr_time = time.time()
        calc_fps = 1.0 / (curr_time - prev_frame_time) if (curr_time - prev_frame_time) > 0 else fps_in
        prev_frame_time = curr_time
        
        # 1. Update Lucas-Kanade Tracker
        lk_vis, lk_stats = lk_tracker.update(frame)
        
        # 2. Update Farneback Dense Tracker
        dense_hsv, quiver_vis, dense_stats = dense_tracker.update(frame)
        
        # Combine Telemetry Stats
        combined_stats = {**lk_stats, **dense_stats}
        combined_stats["frame_index"] = frame_count
        telemetry_logs.append(combined_stats)
        
        # Render 3-Panel Dashboard Montage
        dashboard = render_hud_dashboard(lk_vis, dense_hsv, quiver_vis, combined_stats, fps=calc_fps)
        
        if writer.isOpened():
            writer.write(dashboard)
            
        if is_live:
            cv2.imshow("Optical Flow Motion Tracker - Live HUD", dashboard)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
        else:
            if frame_count % 30 == 0 or frame_count == max_frames:
                print(f"  - Processed Frame {frame_count} | Active Vectors: {lk_stats['num_active_points']} | Avg Speed: {dense_stats['avg_speed_px']} px/f")
            if frame_count >= max_frames:
                break
                
    cap.release()
    writer.release()
    if is_live:
        cv2.destroyAllWindows()
        
    total_time = round(time.time() - start_time, 2)
    avg_fps = round(frame_count / total_time, 1) if total_time > 0 else 0
    
    # Save Telemetry JSON Report
    json_path = os.path.join(output_dir, f"{base_name}_flow_report.json")
    summary = {
        "video_source": base_name,
        "total_frames_processed": frame_count,
        "total_time_seconds": total_time,
        "average_fps": avg_fps,
        "average_motion_speed_px": round(float(np.mean([log["avg_speed_px"] for log in telemetry_logs])), 2) if telemetry_logs else 0,
        "peak_motion_speed_px": round(float(np.max([log["max_speed_px"] for log in telemetry_logs])), 2) if telemetry_logs else 0,
        "output_video": out_video_path
    }
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[OK] Video Processing Complete! Processed {frame_count} frames in {total_time}s ({avg_fps} FPS)")
    print(f"  - Output Video: '{out_video_path}'")
    print(f"  - Telemetry Report: '{json_path}'")
    
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Optical Flow Motion Vector Tracker (Lucas-Kanade & Farneback)."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="input/sample_motion_video.mp4",
        help="Path to video file, folder, or 'camera'/'0' for live webcam."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directory to save output videos and telemetry reports."
    )
    parser.add_argument(
        "-a", "--algorithm", type=str, default="dual",
        choices=["lk", "dense", "dual"],
        help="Optical flow algorithm mode: 'lk' (Lucas-Kanade), 'dense' (Farneback), or 'dual' (3-panel dashboard)."
    )
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="Maximum frames to process for video files (default: 150)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Auto-generate synthetic video if input video file is missing
    if not os.path.exists(args.input) and args.input.lower() not in ["camera", "webcam", "0"]:
        print(f"[!] Input video '{args.input}' not found. Generating synthetic benchmark video...")
        from generate_demo_video import generate_synthetic_motion_video
        args.input = generate_synthetic_motion_video(output_path="input/sample_motion_video.mp4")
        
    print("\n==========================================================")
    print("  [OPT] OPTICAL FLOW MOTION VECTOR TRACKER")
    print("  --------------------------------------------------------")
    print(f"  Input Source   : {args.input}")
    print(f"  Output Dir     : {args.output}")
    print(f"  Algorithm Mode : {args.algorithm.upper()}")
    print("==========================================================")
    
    process_video_stream(
        video_source=args.input,
        output_dir=args.output,
        algorithm=args.algorithm,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
