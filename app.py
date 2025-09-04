#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="基于YOLOv8模型的火灾检测页面",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("YOLOV8火灾检测系统")

# sidebar
st.sidebar.header("模型配置")

# model options
task_type = st.sidebar.selectbox(
    "选择要进行的任务",
    ["检测"]
)

model_type = None
if task_type == "检测":
    model_type = st.sidebar.selectbox(
        "选取模型",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("目前只实现了“检测”功能")

confidence = float(st.sidebar.slider(
    "选取最小置信度", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("请在侧边栏中选择模型")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"无法加载模型。请检查指定的路径: {model_path}")

# image/video options
st.sidebar.header("图片/视频配置")
source_selectbox = st.sidebar.selectbox(
    "选取文件类型",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("目前仅实现了‘图像’和‘视频’作为输入源")