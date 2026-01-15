# 2026赛季视觉组选拔考核 - 项目源码说明

##  项目简介
本项目为 **HITSZ 南工御风·视觉组 2026 赛季选拔考核** 的代码仓库。包含以下两个主要任务的实现代码及实验报告：

1.  **考核任务二：模型训练**
    * 基于 YOLOv11 的目标检测（识别 ball 和 cube）。
    * 包含数据集自动划分脚本与训练启动脚本。
2.  **附加考核：复杂光照下的曝光优化**
    * 基于 OpenCV 和 Flask 开发的 Web 演示系统。
    * 实现了直方图分析、局部对比度检测等过曝诊断算法。
    * 实现了 Gamma 校正、CLAHE（自适应直方图均衡）等图像优化算法。

---

##  环境配置 (Installation)

本项目基于 Python 3.9+ 开发。建议创建虚拟环境运行。

### 1. 安装依赖
请确保根目录下有 `requirements.txt` 文件，运行以下命令安装所需库：

```bash
pip install -r requirements.txt
主要依赖项：

ultralytics (YOLOv11)

opencv-python (图像处理)

flask (Web 演示系统)

matplotlib (绘图)

numpy

 任务说明与运行指南
 1. 考核任务二：YOLO 模型训练
文件结构
split_data.py: 数据集划分脚本（将标注好的数据按 8:2 划分为训练集和验证集）。

train.py: 模型训练启动脚本。

data.yaml: 数据集配置文件。

运行步骤
准备数据：将 X-AnyLabeling 导出的图片和 txt 标签放入 dataset_images 文件夹（或根据脚本修改路径）。

划分数据集：


python split_data.py
执行后，会在 datasets/task2_data 下生成标准的 YOLO 训练格式文件夹。

开始训练：

Bash

python train.py
训练结果（权重文件 best.pt 和图表）将保存在 runs/train/task2_result 目录下。

 2. 附加考核：曝光优化演示系统
这是一个基于 Web 的可视化工具，用于上传图片并实时查看过曝检测结果及 CLAHE 优化效果。

核心文件
app.py: 包含过曝检测算法（直方图、像素阈值、局部对比度等）及 Flask 后端逻辑。

运行步骤
启动 Web 服务：

Bash

python app.py
访问系统： 控制台出现 Running on http://0.0.0.0:5000 后，打开浏览器访问： 👉 http://127.0.0.1:5000

使用功能：

点击“选择文件”上传一张过曝或过暗的图片。

点击“开始检测”。

页面将展示 5 种维度的检测结果，并对比 原图、HE 均衡化 和 CLAHE（推荐） 的效果。

注意：该版本已修复 TypeError: Object of type bool_ is not JSON serializable 报错，可放心使用。

 3. 技术报告 (LaTeX)
本项目的完整技术细节、原理分析及思考题回答已整理为 PDF 报告。

源文件：report.tex

编译方式：使用 XeLaTeX 编译器。

查看报告：直接打开根目录下的 report.pdf 即可查看最终版报告。