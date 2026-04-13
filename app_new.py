# -*- coding: utf-8 -*-
"""
智护安康-脑卒中居家康复平台
Streamlit 一键部署版（自带依赖安装）
"""

# ========== 自动安装依赖（部署时生效，本地运行不影响） ==========
import subprocess
import sys
import importlib

def install_package(package):
    """自动安装指定的Python包"""
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 安装所有需要的依赖
install_package("streamlit")
install_package("opencv-python")
install_package("mediapipe")
install_package("numpy")
install_package("pyttsx3")

# ========== 正式导入库 ==========
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time

# ========== 初始化配置 ==========
# 语音引擎初始化（失败也不影响主程序运行）
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    has_voice = True
except:
    has_voice = False

def voice_prompt(text):
    """语音提示（无报错）"""
    if has_voice:
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass

def calculate_angle(a, b, c):
    """计算三个点之间的夹角"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ========== Streamlit页面设置 ==========
st.set_page_config(
    page_title="智护安康-脑卒中居家康复平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.markdown("1. 点击「开始训练」")
    st.markdown("2. 面对摄像头完成动作")
    st.markdown("3. 系统会自动计数并语音提示")

# 主页面标题
st.title("🧠 智护安康——基于AI的脑卒中居家康复智能平台")

# 初始化MediaPipe姿态检测
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 会话状态初始化
if 'running' not in st.session_state:
    st.session_state.running = False
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# 控制按钮
col1, col2 = st.columns([1, 3])
with col1:
    start_btn = st.checkbox("开始训练", value=st.session_state.running)

if start_btn:
    st.session_state.running = True
    st.session_state.counter = 0
else:
    st.session_state.running = False

# 训练数据显示
st.subheader("实时训练数据")
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

# 摄像头画面占位
frame_placeholder = st.empty()

# ========== 主训练循环 ==========
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    stage = None
    voice_prompt("训练开始，加油！")
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取摄像头画面，请检查设备连接")
            break
        
        # 水平翻转画面（更符合用户视角）
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # 姿态检测
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 绘制关键点
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # 获取关键点坐标
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # 根据动作计算角度
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
            
            # 动作计数逻辑
            if angle > target_angle * 0.9:
                stage = "up"
            if angle < min_angle and stage == "up":
                stage = "down"
                st.session_state.counter += 1
                voice_prompt(f"完成第{st.session_state.counter}次，做得很好")
                
                # 达到目标次数自动停止
                if st.session_state.counter >= target_count:
                    st.session_state.running = False
                    voice_prompt("恭喜你完成训练！")
                    st.success("🎉 训练完成！你太棒了！")
                    st.rerun()
        
        # 显示画面
        frame_placeholder.image(image, channels="BGR", use_column_width=True)
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

else:
    # 未开始训练时的提示画面
    with frame_placeholder.container():
        st.info("点击「开始训练」按钮，开始你的康复练习吧！")
        st.markdown("""
        ### 系统功能说明
        - 实时姿态识别，精准计数
        - 动作语音提示，防止错误姿势
        - 支持多种康复动作选择
        """)