# utils.py
import cv2
import json
import os
import numpy as np
from datetime import datetime
from config import Config


class Utils:
    @staticmethod
    def convert_to_serializable(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
            return obj.item()
        elif isinstance(obj, (list, tuple)):
            return [Utils.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: Utils.convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj

    @staticmethod
    def calculate_speed(position1, position2, time_delta, pixels_per_meter=Config.PIXELS_PER_METER):
        if time_delta == 0:
            return 0.0

        if hasattr(position1[0], 'item'):
            pos1_x = position1[0].item()
            pos1_y = position1[1].item()
        else:
            pos1_x, pos1_y = position1

        if hasattr(position2[0], 'item'):
            pos2_x = position2[0].item()
            pos2_y = position2[1].item()
        else:
            pos2_x, pos2_y = position2

        distance_pixels = ((pos2_x - pos1_x) ** 2 + (pos2_y - pos1_y) ** 2) ** 0.5
        distance_meters = distance_pixels / pixels_per_meter

        speed_ms = distance_meters / time_delta
        speed_kmh = speed_ms * 3.6

        return float(speed_kmh)

    @staticmethod
    def save_violation_image(frame, track_id, violation_type):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"violation_{track_id}_{violation_type}_{timestamp}.jpg"
        filepath = os.path.join(Config.VIOLATIONS_DIR, filename)

        success = cv2.imwrite(filepath, frame)
        if success:
            full_path = os.path.abspath(filepath)
        return filename if success else None

    @staticmethod
    def save_violation_json(violation_data):
        if violation_data is None:
            return

        track_id = violation_data.get('track_id', 'unknown')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violations_{track_id}_{timestamp}.json"
        filepath = os.path.join(Config.JSON_DIR, filename)

        serializable_data = Utils.convert_to_serializable(violation_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump([serializable_data], f, ensure_ascii=False, indent=2)

        full_path = os.path.abspath(filepath)

        return filename

    @staticmethod
    def draw_bounding_box(frame, bbox, track_id, speed=0, violation=""):
        x1, y1, x2, y2 = map(int, bbox)

        if violation == 'wrong_direction':
            color = Config.COLORS['wrong_direction']
        elif violation == 'speeding':
            color = Config.COLORS['violation']
        elif violation == 'illegal_stopping':
            color = Config.COLORS['warning']
        elif violation == 'suspicious_vehicle':
            color = (255, 0, 255)
        else:
            color = Config.COLORS['normal']

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS)

        info_text = f"ID: {track_id}"
        if speed > 5:
            info_text += f" | {speed:.1f} km/h"
        if violation:
            info_text += f" | {violation}"

        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.FONT_THICKNESS)[0]
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + text_size[0] + Config.PADDING, y1), color, -1)
        cv2.putText(frame, info_text, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.COLORS['text'], Config.FONT_THICKNESS)

        return frame

    @staticmethod
    def get_direction_vector(bbox1, bbox2):
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

        direction = (center2[0] - center1[0], center2[1] - center1[1])

        if hasattr(direction[0], 'item'):
            direction = (direction[0].item(), direction[1].item())

        return direction

    @staticmethod
    def add_stats_to_frame(frame, active_vehicles, total_violations):
        stats_text = f"active_vehicles: {active_vehicles} | violations: {total_violations}"

        text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (10, 10), (10 + text_size[0] + 10, 10 + text_size[1] + 10),
                      Config.COLORS['text_bg'], -1)

        cv2.putText(frame, stats_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, Config.COLORS['text'], 2)

        return frame

    @staticmethod
    def show_live_stats():

        try:
            violations_count = len([f for f in os.listdir(Config.VIOLATIONS_DIR) if f.endswith('.jpg')])
            json_count = len([f for f in os.listdir(Config.JSON_DIR) if f.endswith('.json')])

            json_files = [f for f in os.listdir(Config.JSON_DIR) if f.endswith('.json')]
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(Config.JSON_DIR, x)), reverse=True)

            if json_files:
                for i, file in enumerate(json_files[:3]):
                    file_path = os.path.join(Config.JSON_DIR, file)
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%H:%M:%S")
                    print(f"     {i + 1}. {file} ({file_time})")

        except Exception as e:
            print(f" ERROR {e}")