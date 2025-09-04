#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile


def _display_detected_frames(conf, model, st_frame, image):
    """
:param conf (float): 物体检测的置信度阈值。  
:param model (YOLOv8): 包含YOLOv8模型的`YOLOv8`类实例。  
:param st_frame (Streamlit对象): 用于显示检测视频的Streamlit对象。这里可能有些误导因为Streamlit本身不直接处理视频流 但我们可以理解为在Streamlit应用中显示的图像帧。  
:param image (numpy数组): 表示视频帧的numpy数组。  
:return: None  
  
说明 此函数旨在将YOLOv8模型检测到的物体绘制在视频帧上 并通过Streamlit应用进行显示。
它首先使用YOLOv8模型对输入的图像 视频帧 进行物体检测
然后根据设定的置信度阈值筛选出有效的检测结果
并将这些结果绘制到原图像上。最后，虽然函数本身不直接返回处理后的图像
但处理后的图像可以通过Streamlit的某种方式 如`st.image` 在Streamlit应用中显示。  
    """
    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # Predict the objects in the image using YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


@st.cache_resource
def load_model(model_path):
    """
   加载指定路径下的YOLO目标检测模型

   参数：
    model_path (str): YOLO模型文件的路径。

   返回：
    一个YOLO目标检测模型实例。
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_image(conf, model):
    """
   执行上传图像的推理

    param conf: YOLOv8模型的置信度阈值
    param model: 包含YOLOv8模型的YOLOv8类实例
    return: None
    """
    source_img = st.sidebar.file_uploader(
        label="选择一张图片文件",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            # adding the uploaded image to the page with caption
            st.image(
                image=source_img,
                caption="上传图像",
                use_column_width=True
            )

    if source_img:
        if st.button("检测"):
            with st.spinner("运行中..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="被检图像",
                             use_column_width=True)
                    try:
                        with st.expander("检测结果"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("尚未上传图像！")
                        st.write(ex)


def infer_uploaded_video(conf, model):
    """
    执行上传视频的推理

    :param conf: YOLOv8模型的置信度阈值
    :param model: 包含YOLOv8模型的YOLOv8类实例
    :return: None
    """
    source_video = st.sidebar.file_uploader(
        label="选择一个视频文件"
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("检测"):
            with st.spinner("运行中..."):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success, image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,
                                                     model,
                                                     st_frame,
                                                     image
                                                     )
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"Error loading video: {e}")


def infer_uploaded_webcam(conf, model):
    """
    执行针对网络摄像头的推理。

    :param conf: YOLOv8模型的置信度阈值
    :param model: 包含YOLOv8模型的YOLOv8类的一个实例。
    :return: 无（此函数不返回任何值，但会处理视频流并在屏幕上显示结果）
    """
    try:
        flag = st.button(
            label="Stop running"
        )
        vid_cap = cv2.VideoCapture(0)  # local camera
        st_frame = st.empty()
        while not flag:
            success, image = vid_cap.read()
            if success:
                _display_detected_frames(
                    conf,
                    model,
                    st_frame,
                    image
                )
            else:
                vid_cap.release()
                break
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
