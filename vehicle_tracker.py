# vehicle_tracker.py
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from config import Config
from utils import Utils


class VehicleTracker:
    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL)
        self.track_history = defaultdict(lambda: deque(maxlen=Config.TRACK_BUFFER))
        self.vehicle_data = {}

    def detect_vehicles(self, frame):
        try:
            results = self.model.track(
                frame,
                persist=True,
                classes=Config.CLASS_IDS,
                conf=Config.CONFIDENCE_THRESHOLD,
                verbose=False
            )

            return results[0] if results else None
        except Exception as e:
            print(f"ERROR  {e}")
            return None

    def update_tracking(self, results, frame_count, fps):
        current_vehicles = {}

        try:
            if results and results.boxes is not None and results.boxes.id is not None:
                boxes = results.boxes.xywh.cpu()
                track_ids = results.boxes.id.int().cpu().tolist()
                confidences = results.boxes.conf.float().cpu().tolist()

                for box, track_id, confidence in zip(boxes, track_ids, confidences):
                    x = box[0].item() if hasattr(box[0], 'item') else float(box[0])
                    y = box[1].item() if hasattr(box[1], 'item') else float(box[1])
                    w = box[2].item() if hasattr(box[2], 'item') else float(box[2])
                    h = box[3].item() if hasattr(box[3], 'item') else float(box[3])

                    bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

                    self.track_history[track_id].append((float(x), float(y)))

                    if track_id not in self.vehicle_data:
                        self.vehicle_data[track_id] = {
                            'first_seen': int(frame_count),
                            'last_seen': int(frame_count),
                            'positions': deque(maxlen=100),
                            'speed_history': deque(maxlen=10),
                            'stop_start_time': None,
                            'current_speed': 0.0,
                            'violations': [],
                            'direction_history': deque(maxlen=10),
                            'suspicious_start_time': None,
                            'current_violation': '',
                            'violation_start_time': None,
                            'plate_number': None
                        }

                    vehicle = self.vehicle_data[track_id]
                    vehicle['last_seen'] = int(frame_count)
                    vehicle['positions'].append(bbox)

                    if len(vehicle['positions']) >= 2:
                        pos1 = vehicle['positions'][-2]
                        pos2 = vehicle['positions'][-1]
                        center1 = ((pos1[0] + pos1[2]) / 2, (pos1[1] + pos1[3]) / 2)
                        center2 = ((pos2[0] + pos2[2]) / 2, (pos2[1] + pos2[3]) / 2)

                        time_delta = 1.0 / fps
                        speed = Utils.calculate_speed(center1, center2, time_delta)
                        vehicle['current_speed'] = speed
                        vehicle['speed_history'].append(speed)

                    current_violation = self._get_current_violation(vehicle, frame_count, fps)
                    vehicle['current_violation'] = current_violation

                    current_vehicles[track_id] = {
                        'bbox': bbox,
                        'confidence': float(confidence),
                        'speed': float(vehicle['current_speed']),
                        'current_violation': current_violation
                    }

                    if len(vehicle['speed_history']) > 0:
                        avg_speed = sum(vehicle['speed_history']) / len(vehicle['speed_history'])
                        vehicle['current_speed'] = avg_speed

        except Exception as e:
            print(f"ERROR {e}")

        return current_vehicles

    def _get_current_violation(self, vehicle, frame_count, fps):
        current_speed = vehicle.get('current_speed', 0.0)

        if current_speed > Config.SPEED_THRESHOLD_KMH:
            return 'speeding'

        if len(vehicle['direction_history']) > 0:
            last_direction = vehicle['direction_history'][-1]
            dx, dy = last_direction
            if abs(dy) >= 3 and dy < 0:
                return 'wrong_direction'

        if vehicle.get('stop_start_time') is not None:
            stop_duration = (frame_count - vehicle['stop_start_time']) / fps
            if stop_duration > Config.STOP_DURATION_THRESHOLD:
                return 'illegal_stopping'

        if vehicle.get('suspicious_start_time') is not None:
            suspicious_duration = (frame_count - vehicle['suspicious_start_time']) / fps
            if suspicious_duration > Config.SUSPICIOUS_STOP_DURATION:
                return 'suspicious_vehicle'

        return ''

    def get_vehicle_info(self, track_id):
        return self.vehicle_data.get(track_id)

    def cleanup_old_tracks(self, current_frame):
        to_remove = []
        for track_id, data in self.vehicle_data.items():
            if current_frame - data['last_seen'] > Config.MAX_AGE:
                to_remove.append(track_id)

        for track_id in to_remove:
            if track_id in self.vehicle_data:
                del self.vehicle_data[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]

        if to_remove:
            print(f"{len(to_remove)} Vehicle odd")