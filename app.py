import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import math
import warnings
from flask import Flask, render_template, Response

warnings.filterwarnings("ignore")

app = Flask(__name__)

# 1. 설정 및 초기화
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
MAP_WIDTH = 600

BG_COLOR = (40, 40, 40)
ROOM_COLOR = (200, 200, 200)
PATH_COLOR = (100, 100, 100)

ROOMS = {
    'toilet':   (300, 100, 240, 80),
    'room2':    (300, 220, 240, 80),
    'room1':    (300, 340, 240, 80),
    'elevator': (300, 460, 240, 80),
    'home':     (300, 600, 80, 50)
}

MODEL_PATH = 'model/seq_sign_model_final.joblib'
SEQ_LENGTH = 40
ROBOT_IMAGE_PATH = 'raspbot.png'
ROBOT_IMG = None

try:
    img_raw = cv2.imread(ROBOT_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if img_raw is None:
         print(f"⚠ 경고: '{ROBOT_IMAGE_PATH}' 이미지를 찾을 수 없습니다.")
    else:
        ROBOT_IMG = cv2.resize(img_raw, (40, 40))
        if ROBOT_IMG.shape[2] != 4:
            ROBOT_IMG = cv2.cvtColor(ROBOT_IMG, cv2.COLOR_BGR2BGRA)
        print(f"✅ 로봇 이미지 로드 성공: {ROBOT_IMAGE_PATH}")
except Exception as e:
    print(f"❌ 이미지 로드 중 오류 발생: {e}")
    ROBOT_IMG = None

try:
    model = joblib.load(MODEL_PATH)
except:
    print(f"❌ '{MODEL_PATH}' 파일이 없습니다. 경로를 확인하세요.")
    model = None


# 2. 로봇 클래스 & 유틸리티
class Raspbot:
    def __init__(self):
        self.x, self.y = ROOMS['home'][0], ROOMS['home'][1]
        self.target_x, self.target_y = self.x, self.y
        self.speed = 5
        self.radius = 15
        self.status = "Ready"

    def set_target(self, name):
        if name in ROOMS:
            tx, ty, _, _ = ROOMS[name]
            self.target_x, self.target_y = tx, ty
            if name == 'home': self.status = "Returning..."
            else: self.status = f"Moving to {name.upper()}"

    def update(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        if dist > self.speed:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
        else:
            self.x, self.y = self.target_x, self.target_y
            if "Moving" in self.status or "Returning" in self.status:
                self.status = "Arrived"

bot = Raspbot()

def extract_xyz(hand_lms):
    if hand_lms is None: return [0.0] * 63
    out = []
    for lm in hand_lms.landmark:
        out.extend([lm.x, lm.y, lm.z])
    return out

def draw_map_on_image(img):
    cv2.rectangle(img, (0, 0), (MAP_WIDTH, SCREEN_HEIGHT), BG_COLOR, -1)
    cv2.line(img, (300, 100), (300, 600), PATH_COLOR, 10)
    for name, (cx, cy, w, h) in ROOMS.items():
        rect_x = int(cx - w // 2)
        rect_y = int(cy - h // 2)
        if name == 'home': color = (255, 100, 100)
        elif name == 'elevator': color = (100, 255, 255)
        elif name == 'toilet': color = (255, 255, 100)
        else: color = ROOM_COLOR
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + w, rect_y + h), color, -1)
        cv2.rectangle(img, (rect_x, rect_y), (rect_x + w, rect_y + h), (255, 255, 255), 2)
        cv2.putText(img, name.upper(), (rect_x + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

def overlay_transparent(background, overlay, x, y):
    bg_h, bg_w, _ = background.shape
    ol_h, ol_w, _ = overlay.shape
    if x >= bg_w or y >= bg_h or x + ol_w < 0 or y + ol_h < 0: return background
    bg_x1, bg_y1 = max(x, 0), max(y, 0)
    bg_x2, bg_y2 = min(x + ol_w, bg_w), min(y + ol_h, bg_h)
    ol_x1, ol_y1 = max(0, -x), max(0, -y)
    ol_x2, ol_y2 = ol_x1 + (bg_x2 - bg_x1), ol_y1 + (bg_y2 - bg_y1)
    
    overlay_crop = overlay[ol_y1:ol_y2, ol_x1:ol_x2]
    alpha = overlay_crop[:, :, 3] / 255.0
    overlay_rgb = overlay_crop[:, :, :3]
    bg_crop = background[bg_y1:bg_y2, bg_x1:bg_x2]
    
    for c in range(3):
        bg_crop[:, :, c] = (alpha * overlay_rgb[:, :, c] + (1.0 - alpha) * bg_crop[:, :, c])
    background[bg_y1:bg_y2, bg_x1:bg_x2] = bg_crop
    return background

def draw_robot_on_image(img, robot):
    cx, cy = int(robot.x), int(robot.y)
    if ROBOT_IMG is not None:
        h, w, _ = ROBOT_IMG.shape
        overlay_transparent(img, ROBOT_IMG, cx - w//2, cy - h//2)
    else:
        cv2.circle(img, (cx, cy), robot.radius, (50, 50, 255), -1)


# 3. 영상 생성기
def generate_frames():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    seq_buffer = deque(maxlen=SEQ_LENGTH)
    
    global bot
    last_action = "None"
    confidence_disp = 0.0

    while True:
        success, frame = cap.read()
        if not success: break
        
        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        
        left, right = None, None
        if result.multi_hand_landmarks:
            for hl, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                if handedness.classification[0].label == 'Left': left = hl
                else: right = hl
        
        seq_buffer.append(extract_xyz(left) + extract_xyz(right))

        if model and len(seq_buffer) == SEQ_LENGTH:
            input_data = np.array(seq_buffer).flatten().reshape(1, -1)
            probs = model.predict_proba(input_data)[0]
            idx = np.argmax(probs)
            conf = probs[idx]
            action = model.classes_[idx]
            confidence_disp = conf
            if conf > 0.8:
                last_action = action
                if action == 'thankyou': bot.set_target('home')
                elif action in ROOMS: bot.set_target(action)

        draw_map_on_image(canvas)
        bot.update()
        draw_robot_on_image(canvas, bot)

        cam_resized = cv2.resize(frame, (380, 285))
        h, w, _ = cam_resized.shape
        if 50 + h <= SCREEN_HEIGHT and 610 + w <= SCREEN_WIDTH:
            canvas[50:50+h, 610:610+w] = cam_resized

        info_x, info_y = 620, 400
        lines = [f"[ Robot Status ]", f"{bot.status}", "", 
                 f"[ Sign Recognition ]", f"Action: {last_action.upper()}", f"Conf: {confidence_disp*100:.1f}%"]
        
        for i, line in enumerate(lines):
            col = (0, 255, 0) if "Conf" in line and confidence_disp > 0.8 else (255, 255, 255)
            cv2.putText(canvas, line, (info_x, info_y + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

        ret, buffer = cv2.imencode('.jpg', canvas)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# 4. Flask 라우팅
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, port=5000)