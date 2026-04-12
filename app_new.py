import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
from threading import Thread

# 初始化语音引擎
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# 计算两个向量之间的夹角
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# 语音提示防抖
last_voice_time = 0
def voice_prompt(text):
    global last_voice_time
    current_time = time.time()
    if current_time - last_voice_time > 3:
        engine.say(text)
        engine.runAndWait()
        last_voice_time = current_time

# 页面配置
st.set_page_config(page_title="智护安康-脑卒中居家康复平台", layout="wide")

# 侧边栏
with st.sidebar:
    st.header("康复训练设置")
    exercise = st.selectbox(
        "选择康复动作",
        ["手臂上举", "肘关节屈伸", "肩关节外展"]
    )
    target_count = st.slider("目标次数", 1, 50, 10)
    st.markdown("---")
    st.markdown("**操作说明:**")
    st.markdown("点击开始训练，面对摄像头完成动作")

# 主页面
st.title("🧠 智护安康——基于AI的脑卒中居家康复智能平台")

# 初始化MediaPipe（新版写法，解决solutions错误）
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# 自动下载模型文件（第一次运行会自动下载）
import os
import urllib.request
if not os.path.exists('pose_landmarker_lite.task'):
    with st.spinner("正在下载AI模型，请稍候..."):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            "pose_landmarker_lite.task"
        )

# 训练状态
if 'running' not in st.session_state:
    st.session_state.running = False
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# 开始/停止按钮
col1, col2 = st.columns([1, 3])
with col1:
    start_btn = st.checkbox("开始训练", value=st.session_state.running)

if start_btn:
    st.session_state.running = True
    st.session_state.counter = 0
else:
    st.session_state.running = False

# 训练数据显示
st.subheader("训练数据")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("当前动作", exercise)
with col2:
    st.metric("完成次数", st.session_state.counter)
with col3:
    st.metric("目标次数", target_count)
with col4:
    progress = min(100, int(st.session_state.counter / target_count * 100))
    st.metric("完成进度", f"{progress}%")

# 摄像头画面
frame_placeholder = st.empty()

# 主训练循环
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    stage = None
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # 检测姿态
            results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks[0]
                
                # 绘制关键点
                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    results.pose_landmarks[0],
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # 获取左侧身体关键点
                shoulder = [landmarks[11].x, landmarks[11].y]
                elbow = [landmarks[13].x, landmarks[13].y]
                wrist = [landmarks[15].x, landmarks[15].y]
                hip = [landmarks[23].x, landmarks[23].y]
                
                # 计算角度
                if exercise == "手臂上举":
                    angle = calculate_angle(hip, shoulder, wrist)
                    target_angle = 160
                    min_angle = 30
                elif exercise == "肘关节屈伸":
                    angle = calculate_angle(shoulder, elbow, wrist)
                    target_angle = 140
                    min_angle = 60
                elif exercise == "肩关节外展":
                    angle = calculate_angle(hip, shoulder, wrist)
                    target_angle = 90
                    min_angle = 20
                
                # 动作计数
                if angle > target_angle * 0.9:
                    stage = "up"
                if angle < min_angle and stage == "up":
                    stage = "down"
                    st.session_state.counter += 1
                    Thread(target=voice_prompt, args=(f"完成第{st.session_state.counter}次，做得很好",)).start()
                    
                    # 达到目标次数自动停止
                    if st.session_state.counter >= target_count:
                        st.session_state.running = False
                        Thread(target=voice_prompt, args=("恭喜你完成训练！",)).start()
                        st.rerun()
            
            # 显示画面
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
    
    cap.release()
    cv2.destroyAllWindows()