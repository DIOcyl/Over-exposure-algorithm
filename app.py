#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片过曝检测及处理系统 (JSON序列化修复版)
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from flask import Flask, request, jsonify
import base64
from io import BytesIO
import traceback

# 设置中文字体 (兼容Windows/Mac/Linux)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif'] 
plt.rcParams['axes.unicode_minus'] = False

app = Flask(__name__)

class OverexposureDetector:
    def __init__(self, image_path):
        try:
            # 解决中文路径读取问题
            self.image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            if self.image is None:
                raise ValueError("无法解码图像，文件可能损坏")
            
            # 处理4通道(PNG)转3通道
            if len(self.image.shape) == 3 and self.image.shape[2] == 4:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGRA2BGR)
                
            self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError(f"图像读取错误: {str(e)}")
    
    def detect_overexposure_histogram(self, threshold_ratio=0.1):
        """亮度直方图分析"""
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        high_brightness_pixels = np.sum(hist[240:])
        total_pixels = self.gray.size
        ratio = float(high_brightness_pixels / total_pixels) # 强制转 Python float
        return {
            'is_overexposed': bool(ratio > threshold_ratio), # 【修复】强制转 Python bool
            'ratio': ratio, 
            'method': '亮度直方图分析法'
        }
    
    def detect_overexposure_threshold(self, threshold=240):
        """像素阈值统计"""
        _, binary_mask = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        overexposed_pixels = np.sum(binary_mask > 0)
        percentage = float((overexposed_pixels / self.gray.size) * 100)
        return {
            'is_overexposed': bool(percentage > 5), # 【修复】强制转 Python bool
            'percentage': percentage, 
            'method': '像素阈值统计法'
        }
    
    def detect_local_contrast(self, block_size=16, threshold=240):
        """局部对比度分析"""
        h, w = self.gray.shape
        # 为了速度，裁剪到能被 block_size 整除的大小
        h_new = (h // block_size) * block_size
        w_new = (w // block_size) * block_size
        img_trim = self.gray[:h_new, :w_new]
        
        # 利用 reshape 快速分块计算
        blocks = img_trim.reshape(h_new // block_size, block_size, -1, block_size).swapaxes(1, 2)
        block_means = blocks.mean(axis=(2, 3))
        
        overexposed_blocks = np.sum(block_means > threshold)
        total_blocks = block_means.size
        percentage = float((overexposed_blocks / total_blocks) * 100)
        
        return {
            'is_overexposed': bool(percentage > 10), # 【修复】强制转 Python bool
            'percentage': percentage, 
            'method': '局部对比度分析法'
        }

    def detect_brightness_stats(self):
        """均值标准差分析"""
        mean_val = float(np.mean(self.gray))
        std_val = float(np.std(self.gray))
        return {
            'is_overexposed': bool(mean_val > 200 and std_val < 40), # 【修复】强制转 Python bool
            'mean_luminance': mean_val,
            'std_luminance': std_val,
            'method': '亮度均值和标准差检测法'
        }

    def detect_highlight_regions(self, threshold=240, min_area=500):
        """连通区域检测"""
        _, binary_mask = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        # connectivity=8
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        # stats[0] 是背景，跳过。统计面积大于 min_area 的区域数量
        if num_labels > 1:
            large_regions = np.sum(stats[1:, cv2.CC_STAT_AREA] > min_area)
        else:
            large_regions = 0
            
        return {
            'is_overexposed': bool(large_regions > 0), # 【修复】强制转 Python bool
            'large_regions': int(large_regions),       # 【修复】强制转 Python int
            'method': '高光区域检测法'
        }

    def adjust_exposure(self, alpha=0.8, beta=-20):
        return cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
    
    def reduce_highlights(self):
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        L = cv2.equalizeHist(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_Lab2BGR)
    
    def apply_clahe(self):
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2Lab)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        L = clahe.apply(L)
        return cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_Lab2BGR)
    
    def get_histogram_image(self):
        fig = plt.figure(figsize=(8, 4))
        hist = cv2.calcHist([self.gray], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
        plt.title('亮度直方图 (Brightness Histogram)')
        plt.xlabel('像素值')
        plt.ylabel('数量')
        plt.xlim([0, 256])
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close(fig) # 显式关闭 figure 防止内存泄漏
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode()

    def run_all_detections(self):
        return [
            self.detect_overexposure_histogram(),
            self.detect_overexposure_threshold(),
            self.detect_local_contrast(),
            self.detect_brightness_stats(),
            self.detect_highlight_regions()
        ]

@app.route('/')
def index():
    return '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>图片过曝检测系统</title>
    <link href="https://cdn.bootcdn.net/ajax/libs/tailwindcss/2.2.19/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 p-8">
    <div class="max-w-4xl mx-auto bg-white rounded shadow p-6">
        <h1 class="text-2xl font-bold mb-4 text-center">
            📸 图片过曝检测系统 (附加作业演示)
        </h1>
        <form id="uploadForm" class="mb-6">
            <input type="file" id="imageInput" name="image" accept="image/*" class="border p-2 w-full mb-4 rounded">
            <button type="submit" class="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 w-full font-bold">
                🚀 开始检测
            </button>
        </form>
        
        <div id="loading" class="hidden text-center text-blue-600 font-bold my-4 text-xl">
            ⏳ 正在分析图片，请稍候...
        </div>
        <div id="errorBox" class="hidden bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4"></div>

        <div id="results" class="hidden space-y-8">
            <div>
                <h2 class="text-xl font-bold mb-3 border-l-4 border-blue-500 pl-2">🔍 1. 检测结果 (Diagnosis)</h2>
                <div class="grid grid-cols-2 md:grid-cols-3 gap-4" id="detectionCards"></div>
            </div>

            <div>
                <h2 class="text-xl font-bold mb-3 border-l-4 border-blue-500 pl-2">📊 2. 亮度数据 (Histogram)</h2>
                <img id="histImg" class="w-full border rounded shadow-sm">
            </div>

            <div>
                <h2 class="text-xl font-bold mb-3 border-l-4 border-blue-500 pl-2">✨ 3. 优化效果预览 (Optimization)</h2>
                <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
                    <div>
                        <p class="text-sm font-bold text-center mb-1">原始图片</p>
                        <img id="origImg" class="w-full rounded shadow hover:opacity-90 transition">
                    </div>
                    <div>
                        <p class="text-sm font-bold text-center mb-1">线性降低曝光</p>
                        <img id="adjImg" class="w-full rounded shadow hover:opacity-90 transition">
                    </div>
                    <div>
                        <p class="text-sm font-bold text-center mb-1">直方图均衡化(HE)</p>
                        <img id="redImg" class="w-full rounded shadow hover:opacity-90 transition">
                    </div>
                    <div class="relative">
                        <div class="absolute top-0 right-0 bg-green-500 text-white text-xs px-2 py-1 rounded-bl">推荐</div>
                        <p class="text-sm font-bold text-center mb-1 text-green-600">CLAHE (自适应均衡)</p>
                        <img id="claImg" class="w-full rounded shadow border-2 border-green-500 hover:opacity-90 transition">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('imageInput');
            if(fileInput.files.length === 0) return alert("请先选择一张图片！");
            
            // UI重置
            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results').classList.add('hidden');
            document.getElementById('errorBox').classList.add('hidden');

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            try {
                const res = await fetch('/detect', { method: 'POST', body: formData });
                const data = await res.json();
                
                if (!res.ok) throw new Error(data.error || "服务器内部错误");

                // 渲染检测卡片
                const cards = document.getElementById('detectionCards');
                cards.innerHTML = '';
                data.results.forEach(r => {
                    const statusClass = r.is_overexposed ? 'bg-red-50 border-red-200 text-red-700' : 'bg-green-50 border-green-200 text-green-700';
                    const icon = r.is_overexposed ? '⚠️ 过曝' : '✅ 正常';
                    
                    let detail = '';
                    if (r.ratio !== undefined) detail = `占比: ${(r.ratio*100).toFixed(1)}%`;
                    if (r.percentage !== undefined) detail = `占比: ${r.percentage.toFixed(1)}%`;
                    if (r.mean_luminance !== undefined) detail = `均值: ${r.mean_luminance.toFixed(0)}`;
                    if (r.large_regions !== undefined) detail = `高亮区域数: ${r.large_regions}`;

                    cards.innerHTML += `
                        <div class="p-4 border rounded shadow-sm ${statusClass} transition hover:shadow-md">
                            <h4 class="font-bold text-sm mb-1 text-gray-800">${r.method}</h4>
                            <div class="flex justify-between items-center">
                                <span class="font-bold text-lg">${icon}</span>
                                <span class="text-xs opacity-75 bg-white px-1 rounded border">${detail}</span>
                            </div>
                        </div>`;
                });

                // 渲染图片
                document.getElementById('histImg').src = 'data:image/png;base64,' + data.histogram;
                document.getElementById('origImg').src = 'data:image/jpeg;base64,' + data.original;
                document.getElementById('adjImg').src = 'data:image/jpeg;base64,' + data.adjusted;
                document.getElementById('redImg').src = 'data:image/jpeg;base64,' + data.reduced;
                document.getElementById('claImg').src = 'data:image/jpeg;base64,' + data.clahe;

                document.getElementById('results').classList.remove('hidden');
            } catch (err) {
                const errBox = document.getElementById('errorBox');
                errBox.innerText = "❌ 检测失败: " + err.message;
                errBox.classList.remove('hidden');
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        });
    </script>
</body>
</html>
    '''

@app.route('/detect', methods=['POST'])
def detect():
    image_path = "temp_upload_img.jpg"
    try:
        if 'image' not in request.files:
            return jsonify({'error': '未收到图片'}), 400
        
        file = request.files['image']
        file.save(image_path)
        
        detector = OverexposureDetector(image_path)
        results = detector.run_all_detections()
        
        # 辅助函数：转换图片为base64供前端显示
        def to_b64(img):
            _, buf = cv2.imencode('.jpg', img)
            return base64.b64encode(buf).decode()

        response_data = {
            'results': results,
            'histogram': detector.get_histogram_image(),
            'original': to_b64(detector.image),
            'adjusted': to_b64(detector.adjust_exposure()),
            'reduced': to_b64(detector.reduce_highlights()),
            'clahe': to_b64(detector.apply_clahe())
        }
        return jsonify(response_data)

    except Exception as e:
        print("❌ 详细报错信息:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
        
    finally:
        # 清理临时文件
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

if __name__ == '__main__':
    print("正在启动 Web 服务... 请访问 http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)