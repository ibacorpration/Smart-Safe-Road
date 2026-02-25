# config.py
import os
from datetime import datetime

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    VIOLATIONS_DIR = os.path.join(OUTPUT_DIR, "violations")
    JSON_DIR = os.path.join(OUTPUT_DIR, "json")
    PROCESSED_VIDEOS_DIR = os.path.join(OUTPUT_DIR, "processed_videos")

    YOLO_MODEL = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.5
    CLASS_IDS = [2,7]  # car, truck  only

    TRACK_BUFFER = 30
    MATCH_THRESHOLD = 0.8
    MAX_AGE = 100

    SPEED_THRESHOLD_KMH = 190
    STOP_DURATION_THRESHOLD = 5
    SUSPICIOUS_STOP_DURATION = 30
    VIOLATION_COOLDOWN = {
        'speeding': 5,
        'wrong_direction': 999,
        'illegal_stopping': 10,
        'suspicious_vehicle': 30
    }

    CLASS_TO_ARABIC = {
        '0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤',
        '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩',

        'alif': 'ا',
        'baa': 'ب',
        'taa': 'ت',
        'thaa': 'ث',
        'jeem': 'ج',
        'haa': 'ح',
        'khaa': 'خ',
        'daal': 'د',
        'zaal': 'ذ',
        'raa': 'ر',
        'zay': 'ز',
        'seen': 'س',
        'sheen': 'ش',
        'saad': 'ص',
        'daad': 'ض',
        'Taa': 'ط',
        'Thaa': 'ظ',
        'ain': 'ع',
        'ghayn': 'غ',
        'faa': 'ف',
        'qaaf': 'ق',
        'kaaf': 'ك',
        'laam': 'ل',
        'meem': 'م',
        'noon': 'ن',
        'waw': 'و',
        'yaa': 'ي'
    }
    MIN_TRACK_SECONDS = 0.9

    FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    PIXELS_PER_METER = 18

    DISPLAY_ENABLED = True
    SAVE_VIDEO = True
    VIDEO_CODEC = 'mp4v'
    VIDEO_EXTENSION = '.mp4'


    COLORS = {
        'normal': (0, 255, 0),
        'violation': (0, 0, 255),
        'warning': (0, 255, 255),
        'wrong_direction': (0, 165, 255),
        'text_bg': (0, 0, 0),
        'text': (255, 255, 255)
    }

    # تكبير الخط والبوكس
    FONT_SCALE = 0.9
    FONT_THICKNESS = 2
    BOX_THICKNESS = 5
    PADDING = 13

    MAX_FRAME_SKIP = 0
    RESIZE_FACTOR = 1.0

    @classmethod
    def setup_directories(cls):
        directories = [cls.OUTPUT_DIR, cls.VIOLATIONS_DIR, cls.JSON_DIR, cls.PROCESSED_VIDEOS_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    @classmethod
    def get_video_output_path(cls):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"processed_traffic_{timestamp}{cls.VIDEO_EXTENSION}"
        return os.path.join(cls.PROCESSED_VIDEOS_DIR, filename)

Config.setup_directories()