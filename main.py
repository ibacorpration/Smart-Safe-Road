# main.py
import cv2
import time
import os
from datetime import datetime
from config import Config
from vehicle_tracker import VehicleTracker
from violation_detector import ViolationDetector
from utils import Utils
from ultralytics import YOLO

class TrafficMonitoringSystem:
    def __init__(self, video_source=0):
        self.video_source = video_source
        self.cap = None
        self.tracker = VehicleTracker()
        self.violation_detector = ViolationDetector()
        self.frame_count = 0
        self.fps = Config.FPS
        self.total_violations = 0
        self.video_writer = None
        self.plate_detector = YOLO("plate_detector.pt")
        self.ocr_model = YOLO("reader_ocr.pt")

        Config.setup_directories()

    def initialize_video(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f" ERROR {self.video_source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or Config.FPS
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if Config.SAVE_VIDEO:
            video_path = Config.get_video_output_path()
            fourcc = cv2.VideoWriter_fourcc(*Config.VIDEO_CODEC)
            self.video_writer = cv2.VideoWriter(video_path, fourcc, self.fps, (frame_width, frame_height))

        return frame_width, frame_height

    def process_frame(self, frame):
        self.frame_count += 1

        try:
            results = self.tracker.detect_vehicles(frame)
            if results is None:
                return frame, []

            current_vehicles = self.tracker.update_tracking(results, self.frame_count, self.fps)

            violations_in_frame = []
            active_vehicles_count = len(current_vehicles)
            # plate_results = self.plate_detector(frame)[0]
            for track_id, vehicle_info in current_vehicles.items():
                try:
                    vehicle_data = self.tracker.get_vehicle_info(track_id)

                    if vehicle_data and len(vehicle_data['positions']) >= 2:
                        pos1 = vehicle_data['positions'][-2]
                        pos2 = vehicle_data['positions'][-1]
                        direction = Utils.get_direction_vector(pos1, pos2)

                        vehicle_data['direction_history'].append(direction)
                        track_duration = (self.frame_count - vehicle_data['first_seen']) / self.fps

                        if track_duration < Config.MIN_TRACK_SECONDS:
                            continue
                        violations = self.violation_detector.check_all_violations(
                            vehicle_data, track_id, self.frame_count, self.fps, direction
                        )

                        for violation in violations:
                            if violation:
                                image_filename = self._save_violation_with_bbox(
                                    frame, track_id, vehicle_info['bbox'], violation['violation_type'],
                                    vehicle_info['speed']
                                )
                                if image_filename:
                                    violation['image_filename'] = image_filename

                                if vehicle_data.get('plate_number') is None:
                                    plate_results = self.plate_detector(frame)[0]
                                    if len(vehicle_data['positions']) < 5:
                                        continue
                                    if plate_results.boxes is None:
                                        continue

                                    for plate in plate_results.boxes.data.tolist():
                                        px1, py1, px2, py2, pscore, pclass = plate

                                        car_x1, car_y1, car_x2, car_y2 = vehicle_info['bbox']

                                        if px1 > car_x1 and py1 > car_y1 and px2 < car_x2 and py2 < car_y2:

                                            plate_crop = frame[int(py1):int(py2), int(px1):int(px2)]

                                            plate_text, plate_score = self.read_license_plate(plate_crop)

                                            if plate_text:
                                                vehicle_data['plate_number'] = plate_text
                                                break

                                # add plate to violation json
                                plate = vehicle_data.get('plate_number')

                                ordered_violation = {
                                    "track_id": violation.get("track_id"),
                                    "plate_number": plate,
                                    "violation_type": violation.get("violation_type"),
                                    "real_time": violation.get("real_time"),
                                    "description": violation.get("description"),
                                    "image_filename": violation.get("image_filename")
                                }

                                Utils.save_violation_json(ordered_violation)


                                violations_in_frame.append(violation)
                                self.total_violations += 1

                                print(f"🚨 Violation: {violation['description']}")

                        current_violation = vehicle_info.get('current_violation', '')

                        frame = Utils.draw_bounding_box(
                            frame,
                            vehicle_info['bbox'],
                            track_id,
                            vehicle_info['speed'],
                            current_violation
                        )
                except Exception as e:
                    print(f" Error {track_id}: {e}")
                    continue

            frame = Utils.add_stats_to_frame(frame, active_vehicles_count, self.total_violations)

            if self.frame_count % 100 == 0:
                self.tracker.cleanup_old_tracks(self.frame_count)

            return frame, violations_in_frame

        except Exception as e:
            print(f" Error {e}")
            return frame, []

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

    def read_license_plate(self, license_plate_crop):

        if license_plate_crop is None or license_plate_crop.size == 0:
            return None, None

        if len(license_plate_crop.shape) == 2:
            license_plate_crop = cv2.cvtColor(license_plate_crop, cv2.COLOR_GRAY2BGR)

        results = self.ocr_model(license_plate_crop)

        if len(results) == 0 or results[0].boxes is None:
            return None, None

        detections = results[0].boxes
        names = results[0].names

        chars = []

        for box in detections:
            cls_id = int(box.cls[0])
            raw_label = names[cls_id]
            label = Config.CLASS_TO_ARABIC.get(raw_label, raw_label)
            conf = float(box.conf[0])

            if conf < 0.6:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x_center = (x1 + x2) / 2

            chars.append({
                "label": label,
                "conf": conf,
                "x": x_center
            })

        if len(chars) == 0:
            return None, None

        xs = [c["x"] for c in chars]
        plate_mid = (min(xs) + max(xs)) / 2

        arabic_part = []
        digit_part = []

        for c in chars:
            if c["x"] > plate_mid:
                arabic_part.append(c)
            else:
                digit_part.append(c)

        arabic_part = sorted(arabic_part, key=lambda x: x["x"], reverse=True)
        digit_part = sorted(digit_part, key=lambda x: x["x"])

        arabic_text = " ".join([c["label"] for c in arabic_part])
        digit_text = "".join([c["label"] for c in digit_part])

        final_text = arabic_text + " " + digit_text
        avg_score = sum(c["conf"] for c in chars) / len(chars)

        return final_text, avg_score

    def run(self):
        try:
            frame_width, frame_height = self.initialize_video()

            print("\n" + "=" * 60)
            print(" System is running .... \n Vehicles Tracking | Yolo | ByteTrack")
            print("=" * 60)

            start_time = time.time()
            processed_frames = 0
            last_stats_time = time.time()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End tracking ......!")
                    break

                processed_frame, violations = self.process_frame(frame)
                processed_frames += 1

                if Config.SAVE_VIDEO and self.video_writer is not None:
                    self.video_writer.write(processed_frame)

                if Config.DISPLAY_ENABLED:
                    cv2.namedWindow('Vehicles Tracking | Yolo | ByteTrack', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Vehicles Tracking | Yolo | ByteTrack', 1280, 720)
                    cv2.imshow('Vehicles Tracking | Yolo | ByteTrack', processed_frame)

                current_time = time.time()
                if current_time - last_stats_time > 10:
                    Utils.show_live_stats()
                    last_stats_time = current_time

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Stoping Video")
                    break

                if processed_frames % 100 == 0:
                    elapsed_time = time.time() - start_time
                    fps_actual = processed_frames / elapsed_time

        except KeyboardInterrupt:
            print("Stop System !")
        except Exception as e:
            print(f"Error {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.cap:
            self.cap.release()

        if self.video_writer:
            self.video_writer.release()
            print("saved")

        cv2.destroyAllWindows()


        Utils.show_live_stats()

def main():
    video_source = 'egypt_plate3.mp4'
    system = TrafficMonitoringSystem(video_source)
    system.run()

if __name__ == "__main__":
    main()