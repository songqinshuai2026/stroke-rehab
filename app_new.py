# -*- coding: utf-8 -*-
"""
智护安康-脑卒中居家康复平台
终极部署版（自动安装所有依赖，无需requirements.txt）
"""
import subprocess
import sys

# 强制自动安装所有需要的库（云端专用，本地也能用）
def install_and_import(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
        __import__(import_name)

# 按顺序安装所有依赖
install_and_import("streamlit", "streamlit")
install_and_import("opencv-python-headless", "cv2")
install_and_import("mediapipe", "mediapipe")
install_and_import("numpy", "numpy")
install_and_import("pillow", "PIL")

# 现在可以正常导入了
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

# ========== 姿态计算函数 ==========
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# ========== 页面配置 ==========
st.set_page_config(
    page_title="智护安康-脑卒中居家康复平台",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("康复训练设置")
    exercise = st.selectbox("选择康复动作", ["手臂上举", "肘关节屈伸", "肩关节外展"])
    target_count = st.slider("目标次数", 1, 50, 10)
    st.markdown("---")
    st.markdown("### 操作说明")
    st.markdown("1. 勾选「开始训练」按钮")
    st.markdown("2. 允许浏览器访问摄像头")
    st.markdown("3. 面对镜头完成动作，系统自动计数")

# ========== 主页面 ==========
st.title("🧠 智护安康——基于AI的脑卒中居家康复智能平台")

# 会话状态初始化
if 'running' not in st.session_state:
    st.session_state.running = False
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# 控制按钮
col1, _ = st.columns([1, 4])
with col1:
    start_btn = st.checkbox("开始训练", value=st.session_state.running)

if start_btn:
    st.session_state.running = True
    st.session_state.counter = 0
else:
    st.session_state.running = False

# 实时数据显示
st.subheader("训练进度")
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("当前动作", exercise)
with c2: st.metric("完成次数", st.session_state.counter)
with c3: st.metric("目标次数", target_count)
with c4: st.metric("完成进度", f"{min(100, int(st.session_state.counter/target_count*100))}%")

# 画面占位
frame_placeholder = st.empty()

# ========== MediaPipe 初始化 ==========
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ========== 训练主循环（云端兼容） ==========
if st.session_state.running:
    stage = None
    st.info("📷 请在弹出的窗口中允许浏览器访问摄像头")
    
    while st.session_state.running:
        # 调用浏览器摄像头（云端专用方式）
        img_buffer = st.camera_input("摄像头", key="cam", disabled=not st.session_state.running)
        if not img_buffer:
            continue
        
        # 格式转换：PIL → OpenCV
        pil_img = Image.open(img_buffer)
        frame = np.array(pil_img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 1)  # 水平翻转
        
        # 姿态检测
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        results = pose.process(img_rgb)
        img_rgb.flags.writeable = True
        
        # 绘制关键点
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
            # 提取关节点坐标
            lm = results.pose_landmarks.landmark
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            
            # 动作角度判断
            if exercise == "手臂上举":
                angle = calculate_angle(hip, shoulder, wrist)
                target_angle, min_angle = 160, 30
            elif exercise == "肘关节屈伸":
                angle = calculate_angle(shoulder, elbow, wrist)
                target_angle, min_angle = 140, 60
            elif exercise == "肩关节外展":
                angle = calculate_angle(hip, shoulder, wrist)
                target_angle, min_angle = 90, 20
            
            # 计数逻辑
            if angle > target_angle * 0.9:
                stage = "up"
            if angle < min_angle and stage == "up":
                stage = "down"
                st.session_state.counter += 1
                if st.session_state.counter >= target_count:
                    st.session_state.running = False
                    st.success("🎉 恭喜完成所有训练目标！")
                    st.rerun()
        
        # 显示画面
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
    pose.close()
else:
    # 未训练时的提示
    with frame_placeholder.container():
        st.info("点击左侧「开始训练」按钮，开启你的康复训练之旅吧！")
        st.markdown("#### 平台优势")
        st.markdown("- 无需安装软件，浏览器直接使用")
        st.markdown("- AI 实时姿态识别，动作计数精准")
        st.markdown("- 适配居家场景，简单易操作")
