# # import cv2
# # import numpy as np
# # import os
# # from ultralytics import YOLO
# # import pytesseract
# # from collections import defaultdict
# # from scipy.spatial import distance
# #
# # # ==== CONFIG ====
# # video_path = 'cutvideo.mp4'  # Input video
# # goal_model_path = 'goal.pt'       # YOLOv8 for ball + goal net
# # player_model_path = 'yolov8n.pt'  # YOLOv8 for players
# # jersey_model_path = 'best(0-90).pt'  # YOLOv8 for jersey numbers
# #
# # output_folder = 'outputdetection/goal'
# # os.makedirs(output_folder, exist_ok=True)
# #
# # # Tesseract path
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# #
# # # ==== LOAD MODELS ====
# # goal_model = YOLO(goal_model_path)
# # player_model = YOLO(player_model_path)
# # jersey_model = YOLO(jersey_model_path)
# #
# # # ==== OPEN VIDEO ====
# # cap = cv2.VideoCapture(video_path)
# #
# # goal_threshold = 5
# # goal_frame_count = 0
# # ball_trajectory = []
# #
# # # Player tracking variables
# # CONF_THRESHOLD = 0.25
# # player_tracks = {}
# # last_id = 0
# #
# # # ==== Helper Functions ====
# #
# # def preprocess_roi(roi):
# #     roi = cv2.resize(roi, None, fx=2, fy=2)
# #     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
# #     binary = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
# #                                    cv2.THRESH_BINARY_INV, 11, 4)
# #     return binary
# #
# # def enhance_roi(roi):
# #     roi = cv2.resize(roi, None, fx=2, fy=2)
# #     lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
# #     l, a, b = cv2.split(lab)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #     cl = clahe.apply(l)
# #     limg = cv2.merge((cl, a, b))
# #     enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
# #     return enhanced
# #
# # def iou(box1, box2):
# #     x1, y1, x2, y2 = box1
# #     xA, yA, xB, yB = box2
# #     inter_area = max(0, min(x2, xB) - max(x1, xA)) * max(0, min(y2, yB) - max(y1, yA))
# #     box1_area = (x2 - x1) * (y2 - y1)
# #     box2_area = (xB - xA) * (yB - yA)
# #     return inter_area / float(box1_area + box2_area - inter_area + 1e-5)
# #
# # # ==== MAIN LOOP ====
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     ### === Ball & Goal detection ===
# #     results = goal_model(frame, verbose=False)[0]
# #     names = results.names
# #
# #     ball_in_goal = False
# #     ball_center = None
# #     mask_goal_net = None
# #
# #     # Parse masks
# #     if results.masks is not None:
# #         for i, cls_id in enumerate(results.boxes.cls):
# #             cls_name = names[int(cls_id)].lower()
# #             mask = results.masks.data[i].cpu().numpy().astype(np.uint8)
# #             mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
# #
# #             if cls_name == 'goal net':
# #                 mask_goal_net = mask
# #
# #     # Ball box and trajectory
# #     if results.boxes is not None:
# #         for box in results.boxes:
# #             cls_name = names[int(box.cls)].lower()
# #             if "ball" in cls_name:
# #                 x1, y1, x2, y2 = map(int, box.xyxy[0])
# #                 cx = int((x1 + x2) / 2)
# #                 cy = int((y1 + y2) / 2)
# #                 ball_center = (cx, cy)
# #
# #                 ball_trajectory.append(ball_center)
# #                 if len(ball_trajectory) > 50:
# #                     ball_trajectory.pop(0)
# #
# #                 # Check if ball inside goal net
# #                 if mask_goal_net is not None and 0 <= cy < mask_goal_net.shape[0] and 0 <= cx < mask_goal_net.shape[1]:
# #                     if mask_goal_net[cy, cx] == 1:
# #                         ball_in_goal = True
# #                         color = (0, 255, 0)
# #                         cv2.putText(frame, "GOAL DETECTED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
# #                     else:
# #                         color = (0, 0, 255)
# #                 else:
# #                     color = (0, 0, 255)
# #
# #                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
# #                 cv2.circle(frame, ball_center, 7, (255, 0, 0), -1)
# #                 cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
# #
# #     # Draw ball trajectory
# #     for i in range(1, len(ball_trajectory)):
# #         cv2.line(frame, ball_trajectory[i - 1], ball_trajectory[i], (255, 255, 0), 2)
# #
# #     # Overlay goal net mask
# #     if mask_goal_net is not None:
# #         colored_mask = np.zeros_like(frame)
# #         colored_mask[:, :, 1] = mask_goal_net * 255
# #         frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)
# #
# #     if ball_in_goal:
# #         goal_frame_count += 1
# #     else:
# #         goal_frame_count = 0
# #
# #     if goal_frame_count >= goal_threshold:
# #         cv2.putText(frame, "GOAL DETECTED", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
# #
# #     ### === Player & Jersey Detection ===
# #     current_players = []
# #     player_results = player_model(frame, conf=CONF_THRESHOLD)[0]
# #     for box in player_results.boxes.data.tolist():
# #         x1, y1, x2, y2, conf, cls = map(int, box[:6])
# #         current_players.append((x1, y1, x2, y2))
# #
# #     jersey_results = jersey_model(frame, conf=CONF_THRESHOLD)[0]
# #     jersey_numbers = []
# #     for box in jersey_results.boxes.data.tolist():
# #         jx1, jy1, jx2, jy2, conf, cls = map(int, box[:6])
# #         pad = 5
# #         jy1 = max(jy1 - pad, 0)
# #         jx1 = max(jx1 - pad, 0)
# #         jy2 = min(jy2 + pad, frame.shape[0])
# #         jx2 = min(jx2 + pad, frame.shape[1])
# #
# #         roi = frame[jy1:jy2, jx1:jx2]
# #         if roi.size == 0:
# #             continue
# #
# #         processed = preprocess_roi(roi)
# #         enhanced = enhance_roi(roi)
# #         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
# #
# #         custom_config = r'--oem 1 --psm 11 -c tessedit_char_whitelist=0123456789'
# #         texts = [
# #             pytesseract.image_to_string(processed, config=custom_config),
# #             pytesseract.image_to_string(enhanced, config=custom_config),
# #             pytesseract.image_to_string(gray, config=custom_config)
# #         ]
# #
# #         detected_number = ""
# #         for t in texts:
# #             t = ''.join(filter(str.isdigit, t))
# #             if t:
# #                 detected_number = t
# #                 break
# #
# #         if detected_number:
# #             jersey_numbers.append(((jx1, jy1, jx2, jy2), detected_number))
# #
# #     # Match players to jersey numbers and track by ID
# #     new_tracks = {}
# #     for (px1, py1, px2, py2) in current_players:
# #         matched_jersey = None
# #         max_iou = 0
# #         for (jx1, jy1, jx2, jy2), num in jersey_numbers:
# #             iou_val = iou((px1, py1, px2, py2), (jx1, jy1, jx2, jy2))
# #             if iou_val > 0.1 and iou_val > max_iou:
# #                 matched_jersey = num
# #                 max_iou = iou_val
# #
# #         matched_id = None
# #         for pid, data in player_tracks.items():
# #             prev_box = data['bbox']
# #             dist = distance.euclidean(((px1 + px2) / 2, (py1 + py2) / 2),
# #                                       ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2))
# #             if dist < 50:
# #                 matched_id = pid
# #                 break
# #
# #         if matched_id is None:
# #             matched_id = last_id
# #             last_id += 1
# #
# #         if matched_id not in new_tracks:
# #             new_tracks[matched_id] = {
# #                 'bbox': (px1, py1, px2, py2),
# #                 'jersey': matched_jersey if matched_jersey else player_tracks.get(matched_id, {}).get('jersey', "")
# #             }
# #
# #     player_tracks = new_tracks
# #
# #     for pid, data in player_tracks.items():
# #         x1, y1, x2, y2 = data['bbox']
# #         label = f"#{data['jersey']}" if data['jersey'] else "Player"
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
# #         cv2.putText(frame, label, (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
# #
# #     ### === Show result ===
# #     cv2.imshow("Combined Goal & Player Tracking", frame)
# #     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
#
#
#
# import cv2
# import numpy as np
# import os
# from ultralytics import YOLO
# import pytesseract
# from collections import defaultdict
# from scipy.spatial import distance
#
# # ======== CONFIG ========
# video_path = r'D:\dataset\goals\goals (1).mp4'
# #video_path = 'cutvideo.mp4'
# goal_model_path = 'goal.pt'
# foul_model_path = 'yolofoul.pt'
# player_model_path = 'yolov8n.pt'
# jersey_model_path = 'best(0-90).pt'
# output_goal = 'outputdetection/goal'
# output_foul = 'outputdetection/foul'
# os.makedirs(output_goal, exist_ok=True)
# os.makedirs(output_foul, exist_ok=True)
#
# # ======== LOAD MODELS ========
# goal_model = YOLO(goal_model_path)
# foul_model = YOLO(foul_model_path)
# player_model = YOLO(player_model_path)
# jersey_model = YOLO(jersey_model_path)
#
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# target_classes = ['Foul', 'Guilty', 'Victim']
#
# # ======== VIDEO SETUP ========
# cap = cv2.VideoCapture(video_path)
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# # ======== BUFFERS ========
# all_frames = []
# foul_frames = []
# goal_trigger_indices = []
# goal_overlay_frames_remaining = 0
# skip_goal_until = -1
# goal_counter = 1
# foul_counter = 0
# foul_prediction_log = []
#
# frame_idx = 0
#
# # ======== PLAYER TRACKING ========
# player_tracks = {}  # {id: {'bbox': (x1,y1,x2,y2), 'jersey': '7'}}
# last_id = 0
#
# # ======== HELPERS ========
# def preprocess_frame(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
#     sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
#     denoised = cv2.fastNlMeansDenoising(sharpen, h=10)
#     return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
#
# def contains_foul(results):
#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls)
#             cls_name = r.names[cls_id]
#             if cls_name in target_classes:
#                 return True
#     return False
#
# def preprocess_roi(roi):
#     roi = cv2.resize(roi, None, fx=2, fy=2)
#     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
#     binary = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)
#     return binary
#
# def enhance_roi(roi):
#     roi = cv2.resize(roi, None, fx=2, fy=2)
#     lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl,a,b))
#     enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#     return enhanced
#
# def iou(box1, box2):
#     x1, y1, x2, y2 = box1
#     xA, yA, xB, yB = box2
#     inter_area = max(0, min(x2, xB) - max(x1, xA)) * max(0, min(y2, yB) - max(y1, yA))
#     box1_area = (x2 - x1) * (y2 - y1)
#     box2_area = (xB - xA) * (yB - yA)
#     return inter_area / float(box1_area + box2_area - inter_area + 1e-5)
#
# # ======== MAIN LOOP ========
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx += 1
#     all_frames.append(frame.copy())
#     foul_frames.append(frame.copy())
#
#     # ======== GOAL DETECTION ========
#     results_goal = goal_model(frame, verbose=False)[0]
#     names_goal = results_goal.names
#
#     ball_in_goal = False
#     mask_goal_net = None
#
#     if results_goal.masks is not None:
#         for i, cls_id in enumerate(results_goal.boxes.cls):
#             cls_name = names_goal[int(cls_id)].lower()
#             mask = results_goal.masks.data[i].cpu().numpy().astype(np.uint8)
#             mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#             if cls_name == 'goal net':
#                 mask_goal_net = mask
#             elif cls_name == 'ball inside goal net':
#                 ball_in_goal = True
#
#     if results_goal.boxes is not None:
#         for box in results_goal.boxes:
#             cls_name = names_goal[int(box.cls)].lower()
#             if "ball" in cls_name:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cx, cy = int((x1+x2)/2), int((y1+y2)/2)
#                 if mask_goal_net is not None and 0 <= cy < mask_goal_net.shape[0] and 0 <= cx < mask_goal_net.shape[1]:
#                     if mask_goal_net[cy, cx] == 1:
#                         ball_in_goal = True
#
#     if ball_in_goal and frame_idx > skip_goal_until:
#         goal_overlay_frames_remaining = 15
#         goal_trigger_indices.append(frame_idx)
#         skip_goal_until = frame_idx + 150
#
#     if goal_overlay_frames_remaining > 0:
#         cv2.putText(frame, "GOAL DETECTED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
#         goal_overlay_frames_remaining -= 1
#
#     # ======== FOUL DETECTION ========
#     pre_frame = preprocess_frame(frame)
#     foul_results = foul_model(pre_frame, verbose=False)
#     detected_foul = contains_foul(foul_results)
#     foul_prediction_log.append(detected_foul)
#     if len(foul_prediction_log) >= 50 and all(foul_prediction_log[-50:]):
#         cv2.putText(frame, "FOUL DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
#
#     # ======== PLAYER & JERSEY DETECTION ========
#     CONF_THRESHOLD = 0.25
#     current_players = []
#     player_results = player_model(frame, conf=CONF_THRESHOLD)[0]
#     for box in player_results.boxes.data.tolist():
#         x1, y1, x2, y2, conf, cls = map(int, box[:6])
#         current_players.append((x1, y1, x2, y2))
#
#     jersey_results = jersey_model(frame, conf=CONF_THRESHOLD)[0]
#     jersey_numbers = []
#     for box in jersey_results.boxes.data.tolist():
#         jx1, jy1, jx2, jy2, conf, cls = map(int, box[:6])
#         pad = 5
#         jy1 = max(jy1 - pad, 0)
#         jx1 = max(jx1 - pad, 0)
#         jy2 = min(jy2 + pad, frame.shape[0])
#         jx2 = min(jx2 + pad, frame.shape[1])
#         roi = frame[jy1:jy2, jx1:jx2]
#         if roi.size == 0:
#             continue
#         processed = preprocess_roi(roi)
#         enhanced = enhance_roi(roi)
#         gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         custom_config = r'--oem 1 --psm 11 -c tessedit_char_whitelist=0123456789'
#         texts = [
#             pytesseract.image_to_string(processed, config=custom_config),
#             pytesseract.image_to_string(enhanced, config=custom_config),
#             pytesseract.image_to_string(gray, config=custom_config)
#         ]
#         detected_number = ""
#         for t in texts:
#             t = ''.join(filter(str.isdigit, t))
#             if t:
#                 detected_number = t
#                 break
#         if detected_number:
#             jersey_numbers.append(((jx1, jy1, jx2, jy2), detected_number))
#
#     # ======== TRACKING ========
#     new_tracks = {}
#     for (px1, py1, px2, py2) in current_players:
#         matched_jersey = None
#         max_iou = 0
#         for (jx1, jy1, jx2, jy2), num in jersey_numbers:
#             iou_val = iou((px1, py1, px2, py2), (jx1, jy1, jx2, jy2))
#             if iou_val > 0.1 and iou_val > max_iou:
#                 matched_jersey = num
#                 max_iou = iou_val
#         matched_id = None
#         for pid, data in player_tracks.items():
#             prev_box = data['bbox']
#             dist = distance.euclidean(((px1 + px2) / 2, (py1 + py2) / 2),
#                                       ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2))
#             if dist < 50:
#                 matched_id = pid
#                 break
#         if matched_id is None:
#             matched_id = last_id
#             last_id += 1
#         if matched_id not in new_tracks:
#             new_tracks[matched_id] = {
#                 'bbox': (px1, py1, px2, py2),
#                 'jersey': matched_jersey if matched_jersey else player_tracks.get(matched_id, {}).get('jersey', "")
#             }
#     player_tracks = new_tracks
#
#     for pid, data in player_tracks.items():
#         x1, y1, x2, y2 = data['bbox']
#         label = f"#{data['jersey']}" if data['jersey'] else "Player"
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#
#     # ======== DISPLAY ========
#     cv2.imshow("Match Analysis", frame)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# # ======== SAVE GOAL CLIPS ========
# for idx in goal_trigger_indices:
#     start_idx = max(0, idx - 50)
#     end_idx = min(idx + 150 + 15, len(all_frames))
#     clip_frames = all_frames[start_idx:end_idx]
#     for i in range(15):
#         overlay_idx = idx - start_idx + i
#         if overlay_idx < len(clip_frames):
#             cv2.putText(clip_frames[overlay_idx], "GOAL DETECTED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
#     clip_path = os.path.join(output_goal, f'goal_clip_{goal_counter}.mp4')
#     out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#     for f in clip_frames:
#         out.write(f)
#     out.release()
#     print(f"✅ Goal Saved: {clip_path}")
#     goal_counter += 1
#
# # ======== SAVE FOUL CLIPS ========
# i = 0
# total_frames = len(foul_prediction_log)
# while i < total_frames - 50:
#     if all(foul_prediction_log[i:i+50]):
#         start_idx = max(0, i - 10)
#         end_idx = min(total_frames, i + 100)
#         clip_frames = foul_frames[start_idx:end_idx]
#         clip_path = os.path.join(output_foul, f'foul_clip_{foul_counter}.mp4')
#         out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#         for f in clip_frames:
#             out.write(f)
#         out.release()
#         print(f"✅ Foul Saved: {clip_path}")
#         foul_counter += 1
#         i = end_idx
#     else:
#         i += 1
#
# print(f"\n🎥 Total goal clips saved: {goal_counter - 1}")
# print(f"🎥 Total foul clips saved: {foul_counter}")


import cv2
import numpy as np
import os
from ultralytics import YOLO
import pytesseract
from collections import defaultdict
from scipy.spatial import distance

# ========== CONFIG ==========
video_path = r'D:\dataset\goals\goals (1).mp4'  # Your video
goal_model_path = 'goal.pt'
foul_model_path = 'yolofoul.pt'
player_model_path = 'yolov8n.pt'
jersey_model_path = 'best(0-90).pt'

output_goal = 'outputdetection/goal'
output_foul = 'outputdetection/foul'
os.makedirs(output_goal, exist_ok=True)
os.makedirs(output_foul, exist_ok=True)

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ========== LOAD MODELS ==========
goal_model = YOLO(goal_model_path)
foul_model = YOLO(foul_model_path)
player_model = YOLO(player_model_path)
jersey_model = YOLO(jersey_model_path)

target_classes = ['Foul', 'Guilty', 'Victim']

CONF_THRESHOLD = 0.25

# ========== INIT VIDEO ==========
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Buffers
all_frames = []
foul_frames = []
foul_prediction_log = []
goal_trigger_indices = []
goal_overlay_frames_remaining = 0
skip_goal_until = -1
goal_counter = 1
foul_counter = 0

frame_idx = 0

# Player Tracking
player_tracks = {}
last_id = 0

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
    denoised = cv2.fastNlMeansDenoising(sharpen, h=10)
    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

def preprocess_roi(roi):
    roi = cv2.resize(roi, None, fx=2, fy=2)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
    binary = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)
    return binary

def enhance_roi(roi):
    roi = cv2.resize(roi, None, fx=2, fy=2)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    xA, yA, xB, yB = box2
    inter_area = max(0, min(x2, xB) - max(x1, xA)) * max(0, min(y2, yB) - max(y1, yA))
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (xB - xA) * (yB - yA)
    return inter_area / float(box1_area + box2_area - inter_area + 1e-5)

def contains_foul(results):
    for r in results:
        for box in r.boxes:
            cls_name = r.names[int(box.cls)]
            if cls_name in target_classes:
                return True
    return False

# ========== MAIN LOOP ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    all_frames.append(frame.copy())
    foul_frames.append(frame.copy())

    # ----- GOAL DETECTION -----
    results_goal = goal_model(frame, verbose=False)[0]
    names_goal = results_goal.names

    ball_in_goal = False
    mask_goal_net = None

    if results_goal.masks is not None:
        for i, cls_id in enumerate(results_goal.boxes.cls):
            cls_name = names_goal[int(cls_id)].lower()
            mask = results_goal.masks.data[i].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            if cls_name == 'goal net':
                mask_goal_net = mask
            elif cls_name == 'ball inside goal net':
                ball_in_goal = True

    if results_goal.boxes is not None:
        for box in results_goal.boxes:
            cls_name = names_goal[int(box.cls)].lower()
            if "ball" in cls_name:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if mask_goal_net is not None and mask_goal_net[cy, cx] == 1:
                    ball_in_goal = True

    if ball_in_goal and frame_idx > skip_goal_until:
        goal_overlay_frames_remaining = 15
        goal_trigger_indices.append(frame_idx)
        skip_goal_until = frame_idx + 150

    if goal_overlay_frames_remaining > 0:
        cv2.putText(frame, "GOAL DETECTED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        goal_overlay_frames_remaining -= 1

    # ----- FOUL DETECTION -----
    pre_frame = preprocess_frame(frame)
    foul_results = foul_model(pre_frame, verbose=False)
    detected_foul = contains_foul(foul_results)
    foul_prediction_log.append(detected_foul)

    if len(foul_prediction_log) >= 50 and all(foul_prediction_log[-50:]):
        cv2.putText(frame, "FOUL DETECTED", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # ----- PLAYER + JERSEY DETECTION -----
    current_players = []
    player_results = player_model(frame, conf=CONF_THRESHOLD)[0]
    for box in player_results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = map(int, box[:6])
        current_players.append((x1, y1, x2, y2))

    jersey_results = jersey_model(frame, conf=CONF_THRESHOLD)[0]
    jersey_numbers = []
    for box in jersey_results.boxes.data.tolist():
        jx1, jy1, jx2, jy2, conf, cls = map(int, box[:6])
        pad = 5
        jy1, jx1 = max(jy1 - pad, 0), max(jx1 - pad, 0)
        jy2, jx2 = min(jy2 + pad, frame.shape[0]), min(jx2 + pad, frame.shape[1])
        roi = frame[jy1:jy2, jx1:jx2]
        if roi.size == 0:
            continue
        processed = preprocess_roi(roi)
        enhanced = enhance_roi(roi)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        custom_config = r'--oem 1 --psm 11 -c tessedit_char_whitelist=0123456789'
        texts = [
            pytesseract.image_to_string(processed, config=custom_config),
            pytesseract.image_to_string(enhanced, config=custom_config),
            pytesseract.image_to_string(gray, config=custom_config)
        ]
        detected_number = ""
        for t in texts:
            t = ''.join(filter(str.isdigit, t))
            if t:
                detected_number = t
                break
        if detected_number:
            jersey_numbers.append(((jx1, jy1, jx2, jy2), detected_number))

    new_tracks = {}
    for (px1, py1, px2, py2) in current_players:
        matched_jersey = None
        max_iou = 0
        for (jx1, jy1, jx2, jy2), num in jersey_numbers:
            iou_val = iou((px1, py1, px2, py2), (jx1, jy1, jx2, jy2))
            if iou_val > 0.1 and iou_val > max_iou:
                matched_jersey = num
                max_iou = iou_val

        matched_id = None
        for pid, data in player_tracks.items():
            prev_box = data['bbox']
            dist = distance.euclidean(((px1 + px2) / 2, (py1 + py2) / 2),
                                      ((prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2))
            if dist < 50:
                matched_id = pid
                break

        if matched_id is None:
            matched_id = last_id
            last_id += 1

        if matched_id not in new_tracks:
            new_tracks[matched_id] = {
                'bbox': (px1, py1, px2, py2),
                'jersey': matched_jersey if matched_jersey else player_tracks.get(matched_id, {}).get('jersey', "")
            }

    player_tracks = new_tracks

    for pid, data in player_tracks.items():
        x1, y1, x2, y2 = data['bbox']
        label = f"#{data['jersey']}" if data['jersey'] else "Player"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show output
    cv2.imshow("Match Analysis", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# ========== SAVE GOAL CLIPS ==========
for idx in goal_trigger_indices:
    start_idx = max(0, idx - 50)
    end_idx = min(idx + 150 + 15, len(all_frames))
    clip_frames = all_frames[start_idx:end_idx]
    for i in range(15):
        overlay_idx = idx - start_idx + i
        if overlay_idx < len(clip_frames):
            cv2.putText(clip_frames[overlay_idx], "GOAL DETECTED", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    clip_path = os.path.join(output_goal, f'goal_clip_{goal_counter}.mp4')
    out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for f in clip_frames:
        out.write(f)
    out.release()
    print(f"✅ Goal Saved: {clip_path}")
    goal_counter += 1

# ========== SAVE FOUL CLIPS ==========
i = 0
total_frames = len(foul_prediction_log)
while i < total_frames - 50:
    if all(foul_prediction_log[i:i+50]):
        start_idx = max(0, i - 10)
        end_idx = min(total_frames, i + 100)
        clip_frames = foul_frames[start_idx:end_idx]
        clip_path = os.path.join(output_foul, f'foul_clip_{foul_counter}.mp4')
        out = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for f in clip_frames:
            out.write(f)
        out.release()
        print(f"✅ Foul Saved: {clip_path}")
        foul_counter += 1
        i = end_idx
    else:
        i += 1

print(f"\n🎥 Total goal clips saved: {goal_counter - 1}")
print(f"🎥 Total foul clips saved: {foul_counter}")
