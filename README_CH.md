<div align="center">

# YOLOv8 Streamlit 火焰检测应用

  <p>
    <a align="center" href="https://ultralytics.com/yolov8" target="_blank">
      <img width="50%" src="pic_bed/banner-yolov8.png"></a>
  </p>

<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 引用"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker 拉取次数"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="在 Gradient 上运行"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="在 Kaggle 中打开"></a>
  </div>
  <br>
</div>

## 项目简介
本仓库提供了一个基于 [YOLOv8](https://github.com/ultralytics/ultralytics) 的用户友好交互界面，该界面由 [Streamlit](https://github.com/streamlit/streamlit) 驱动。该项目可作为您开发自己项目时的参考资源。

## 功能特点
- **功能一**: 目标检测任务（专注于火焰检测）
- **功能二**: 多种检测模型选择，支持 `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- **功能三**: 多种输入格式，支持 `图像`, `视频`, `摄像头实时输入`

## 安装指南

### 创建新的 Conda 环境
```bash
# 创建环境
conda create -n yolov8 python=3.9 -y

# 激活环境
conda activate yolov8
```

### 克隆仓库
```bash
git clone https://github.com/JackDance/YOLOv8-streamlit-app
cd YOLOv8-streamlit-app
```

### 安装依赖包
```bash
# YOLOv8 依赖
pip install ultralytics

# Streamlit 依赖
pip install streamlit

# 可选：图像处理相关依赖
pip install opencv-python pillow
```

### 下载预训练的 YOLOv8 检测权重
创建一个名为 `weights` 的目录，然后在其中创建 `detection` 子目录，并将下载的 YOLOv8 目标检测权重文件保存到此目录中。权重文件可从下表下载：

| 模型                                                                                | 尺寸<br><sup>(像素) | mAP<sup>val<br>50-95 | 速度<br><sup>CPU ONNX<br>(ms) | 速度<br><sup>A100 TensorRT<br>(ms) | 参数量<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | ------------------- | -------------------- | ----------------------------- | ---------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                 | 37.3                 | 80.4                          | 0.99                               | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                 | 44.9                 | 128.4                         | 1.20                               | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                 | 50.2                 | 234.7                         | 1.83                               | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                 | 52.9                 | 375.2                         | 2.39                               | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                 | 53.9                 | 479.1                         | 3.53                               | 68.2               | 257.8             |

## 运行应用
```bash
streamlit run app.py
```
运行后会自动启动 Streamlit 服务器并在默认浏览器中打开应用页面。

## 常见问题
1. **权重文件放置**：确保权重文件正确放置在 `weights/detection/` 目录中
2. **依赖问题**：如果遇到依赖冲突，建议使用全新的 Conda 环境
3. **摄像头访问**：首次使用摄像头功能时，浏览器会请求摄像头访问权限，请点击允许

## 待开发功能
- [ ] 添加 `目标跟踪` 功能
- [ ] 添加 `图像分类` 功能
- [ ] 添加 `姿态估计` 功能

***
