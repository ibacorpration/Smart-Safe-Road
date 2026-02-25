# violation_detector.py
import json
import numpy as np
from datetime import datetime
from config import Config
from utils import Utils
import cv2


class ViolationDetector:
    def __init__(self):
        self.road_direction = None
        self.violation_count = 0

    def _prepare_violation_data(self, violation_info):
        if violation_info is None:
            return None

        prepared_data = {}
        for key, value in violation_info.items():
            prepared_data[key] = Utils.convert_to_serializable(value)

        return prepared_data

    def _violation_exists(self, vehicle_data, violation_type, time_window_seconds):
        current_time = vehicle_data.get('last_seen', 0)
        fps = Config.FPS
        time_window_frames = time_window_seconds * fps

        for violation in vehicle_data.get('violations', []):
            if (violation['type'] == violation_type and
                    current_time - violation.get('frame_count', 0) < time_window_frames):
                return True
        return False

    def _save_violation_with_bbox(self, frame, track_id, bbox, violation_type, speed=0):
        violation_frame = frame.copy()

        if violation_type == 'wrong_direction':
            color = Config.COLORS['wrong_direction']
        else:
            color = Config.COLORS['violation']

        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(violation_frame, (x1, y1), (x2, y2), color, Config.BOX_THICKNESS)

        info_text = f"ID: {track_id} | {violation_type}"
        if speed > 5:
            info_text += f" | {speed:.1f} km/h"

        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.FONT_THICKNESS)[0]
        cv2.rectangle(violation_frame, (x1, y1 - 35), (x1 + text_size[0] + Config.PADDING, y1), color, -1)
        cv2.putText(violation_frame, info_text, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, Config.FONT_SCALE, Config.COLORS['text'], Config.FONT_THICKNESS)

        return Utils.save_violation_image(violation_frame, track_id, violation_type)

    def detect_speeding(self, vehicle_data, current_speed, track_id, frame_count, fps):
        if hasattr(current_speed, 'item'):
            current_speed = current_speed.item()

        current_speed = float(current_speed)

        if current_speed > Config.SPEED_THRESHOLD_KMH:
            violation_time = float(frame_count / fps)
            real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            violation_info = {
                'track_id': int(track_id),
                'violation_type': 'speeding',
                # 'violation_time': violation_time,
                'real_time': real_time,
                'speed': current_speed,
                'description': f'المركبة {track_id} تجاوزت السرعة المسموح بها: {current_speed:.1f} كم/ساعة'
            }

            if not self._violation_exists(vehicle_data, 'speeding', Config.VIOLATION_COOLDOWN['speeding']):
                vehicle_data['violations'].append({
                    'type': 'speeding',
                    # 'time': violation_time,
                    'real_time': real_time,
                    'speed': current_speed,
                    'frame_count': int(frame_count)
                })
                return self._prepare_violation_data(violation_info)

        return None

    def detect_wrong_direction(self, vehicle_data, current_direction, track_id, frame_count, fps):
        dx, dy = current_direction
        if abs(dy) < 3:
            return None

        is_wrong_direction = dy < 0

        if is_wrong_direction:
            already_violated = any(v['type'] == 'wrong_direction' for v in vehicle_data.get('violations', []))
            if not already_violated:
                violation_time = float(frame_count / fps)
                real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # الوقت الحقيقي

                violation_info = {
                    'track_id': int(track_id),
                    'violation_type': 'wrong_direction',
                    # 'violation_time': violation_time,
                    'real_time': real_time,
                    'description': f'المركبة {track_id} تسير عكس الاتجاه'
                }

                vehicle_data['violations'].append({
                    'type': 'wrong_direction',
                    # 'time': violation_time,
                    'real_time': real_time,
                    'frame_count': int(frame_count)
                })
                return self._prepare_violation_data(violation_info)
        return None

    def detect_illegal_stopping(self, vehicle_data, current_speed, track_id, frame_count, fps):
        if hasattr(current_speed, 'item'):
            current_speed = current_speed.item()

        current_speed = float(current_speed)

        if current_speed < 1.0:
            if vehicle_data['stop_start_time'] is None:
                vehicle_data['stop_start_time'] = int(frame_count)
            else:
                stop_duration = float((frame_count - vehicle_data['stop_start_time']) / fps)

                if stop_duration > Config.STOP_DURATION_THRESHOLD:
                    violation_time = float(frame_count / fps)
                    real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    violation_info = {
                        'track_id': int(track_id),
                        'violation_type': 'illegal_stopping',
                        # 'violation_time': violation_time,
                        'real_time': real_time,
                        'duration': float(stop_duration),
                        'description': f'المركبة {track_id} متوقفة في منطقة ممنوعة لمدة {stop_duration:.1f} ثانية'
                    }

                    if not self._violation_exists(vehicle_data, 'illegal_stopping',
                                                  Config.VIOLATION_COOLDOWN['illegal_stopping']):
                        vehicle_data['violations'].append({
                            'type': 'illegal_stopping',
                            # 'time': violation_time,
                            'real_time': real_time,
                            'duration': float(stop_duration),
                            'frame_count': int(frame_count)
                        })
                        return self._prepare_violation_data(violation_info)
        else:
            vehicle_data['stop_start_time'] = None

        return None

    def detect_suspicious_vehicle(self, vehicle_data, current_speed, track_id, frame_count, fps):
        if hasattr(current_speed, 'item'):
            current_speed = current_speed.item()

        current_speed = float(current_speed)

        if current_speed < 0.5:
            if vehicle_data.get('suspicious_start_time') is None:
                vehicle_data['suspicious_start_time'] = int(frame_count)
            else:
                suspicious_duration = float((frame_count - vehicle_data['suspicious_start_time']) / fps)

                if suspicious_duration > Config.SUSPICIOUS_STOP_DURATION:
                    violation_time = float(frame_count / fps)
                    real_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    violation_info = {
                        'track_id': int(track_id),
                        'violation_type': 'suspicious_vehicle',
                        # 'violation_time': violation_time,
                        'real_time': real_time,
                        'duration': float(suspicious_duration),
                        'description': f'المركبة {track_id} ثابتة لمدة {suspicious_duration:.1f} ثانية - حالة مشبوهة'
                    }

                    if not self._violation_exists(vehicle_data, 'suspicious_vehicle',
                                                  Config.VIOLATION_COOLDOWN['suspicious_vehicle']):
                        vehicle_data['violations'].append({
                            'type': 'suspicious_vehicle',
                            # 'time': violation_time,
                            'real_time': real_time,  # إضافة الوقت الحقيقي
                            'duration': float(suspicious_duration),
                            'frame_count': int(frame_count)
                        })
                        return self._prepare_violation_data(violation_info)
        else:
            vehicle_data['suspicious_start_time'] = None

        return None

    def check_all_violations(self, vehicle_data, track_id, frame_count, fps, current_direction=(1, 0)):
        violations_detected = []
        current_speed = vehicle_data.get('current_speed', 0.0)

        speeding_violation = self.detect_speeding(vehicle_data, current_speed, track_id, frame_count, fps)
        if speeding_violation:
            violations_detected.append(speeding_violation)

        # فحص الاتجاه الخاطئ
        wrong_dir_violation = self.detect_wrong_direction(vehicle_data, current_direction, track_id, frame_count, fps)
        if wrong_dir_violation:
            violations_detected.append(wrong_dir_violation)

        # فحص الوقوف الممنوع
        stopping_violation = self.detect_illegal_stopping(vehicle_data, current_speed, track_id, frame_count, fps)
        if stopping_violation:
            violations_detected.append(stopping_violation)

        # فحص المركبة المشبوهة
        suspicious_violation = self.detect_suspicious_vehicle(vehicle_data, current_speed, track_id, frame_count, fps)
        if suspicious_violation:
            violations_detected.append(suspicious_violation)

        return violations_detected