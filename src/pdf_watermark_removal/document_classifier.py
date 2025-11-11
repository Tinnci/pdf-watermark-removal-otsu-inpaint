"""Document type classifier for automatic parameter optimization."""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass


class DocumentType(Enum):
    """Document classification types."""

    ELECTRONIC = "electronic"  # 电子文档：颜色离散、文字纯黑、边缘锐利
    SCANNED = "scanned"  # 扫描文档：颜色连续、噪点多、边缘模糊
    MIXED = "mixed"  # 混合类型：需要保守参数


@dataclass
class ClassificationResult:
    """Classification results with metrics."""

    doc_type: DocumentType
    confidence: float
    metrics: dict


class DocumentClassifier:
    """Intelligently classifies document type based on visual features."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def classify(self, image_rgb):
        """
        基于多维度特征对文档进行分类

        Args:
            image_rgb: 第一页图像(RGB格式)

        Returns:
            ClassificationResult对象
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # 1. 颜色离散度分析（电子文档颜色少，扫描文档颜色连续）
        color_discreteness = self._analyze_color_discreteness(image_rgb)

        # 2. 文字灰度集中度（电子文档文字纯黑且集中）
        text_gray_concentration = self._analyze_text_gray_distribution(gray)

        # 3. 边缘锐度分析（电子文档边缘锐利）
        edge_sharpness = self._analyze_edge_sharpness(gray)

        # 4. 噪点水平检测（扫描文档噪点多）
        noise_level = self._analyze_noise_level(gray)

        # 综合评分
        metrics = {
            "color_discreteness": color_discreteness,
            "text_concentration": text_gray_concentration,
            "edge_sharpness": edge_sharpness,
            "noise_level": noise_level,
        }

        doc_type, confidence = self._decide_type(metrics)

        if self.verbose:
            print(
                f"[DocumentClassifier] Type: {doc_type.value}, "
                f"Confidence: {confidence:.1f}%"
            )
            for key, val in metrics.items():
                print(f"  {key}: {val:.1f}")

        return ClassificationResult(doc_type, confidence, metrics)

    def _analyze_color_discreteness(self, image_rgb, sample_size=1000):
        """
        分析颜色离散度：电子文档通常只有几种离散颜色

        Returns: 离散度分数 (0-100, 越高越离散)
        """
        # 下采样加速计算
        h, w = image_rgb.shape[:2]
        if h * w > sample_size * 4:
            scale = np.sqrt(sample_size / (h * w))
            img_small = cv2.resize(image_rgb, None, fx=scale, fy=scale)
        else:
            img_small = image_rgb

        # 统计唯一颜色数量（量化减少噪点影响）
        quantized = (img_small // 8) * 8  # 量化到32级
        unique_colors = np.unique(quantized.reshape(-1, 3), axis=0)

        # 电子文档：通常<50种颜色；扫描文档：>200种
        discreteness = min(100, max(0, 100 - len(unique_colors) / 2))
        return discreteness

    def _analyze_text_gray_distribution(self, gray):
        """
        分析文字灰度分布：电子文档文字集中在灰度0附近

        Returns: 集中度分数 (0-100, 越高越集中)
        """
        # 检测暗色区域（潜在文字）
        dark_mask = gray < 100
        if np.count_nonzero(dark_mask) < 100:
            return 50  # 无文字，中性分数

        dark_pixels = gray[dark_mask]

        # 计算主要暗色峰值的标准差
        hist, bins = np.histogram(dark_pixels, bins=20, range=(0, 100))
        peak_bin = np.argmax(hist)
        peak_pixels = dark_pixels[
            (dark_pixels >= bins[peak_bin]) & (dark_pixels < bins[peak_bin + 1])
        ]

        std_dev = np.std(peak_pixels) if len(peak_pixels) > 10 else 255

        # 电子文档：std_dev < 15；扫描文档：std_dev > 30
        concentration = max(0, 100 - std_dev * 2.5)
        return concentration

    def _analyze_edge_sharpness(self, gray):
        """
        分析边缘锐度：电子文档边缘锐利，扫描文档模糊

        Returns: 锐度分数 (0-100, 越高越锐利)
        """
        # 使用拉普拉斯方差作为锐度指标
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()

        # 归一化：电子文档>1000，扫描文档<300
        normalized = min(100, sharpness / 10)
        return normalized

    def _analyze_noise_level(self, gray):
        """
        分析噪点水平：扫描文档噪点多

        Returns: 噪点分数 (0-100, 越高噪点越少)
        """
        # 高频成分分析
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        diff = np.abs(gray.astype(int) - denoised.astype(int))
        noise_ratio = np.mean(diff) / 255

        # 电子文档：noise_ratio < 0.02；扫描文档：>0.08
        cleanness = max(0, 100 - noise_ratio * 1000)
        return cleanness

    def _decide_type(self, metrics):
        """
        基于多特征综合决策文档类型

        Returns: (DocumentType, confidence)
        """
        # 加权评分
        electronic_score = (
            metrics["color_discreteness"] * 0.3
            + metrics["text_concentration"] * 0.3
            + metrics["edge_sharpness"] * 0.2
            + metrics["noise_level"] * 0.2
        )

        # 扫描文档是电子文档的反面
        scanned_score = 100 - electronic_score

        if electronic_score > 70 and scanned_score < 30:
            return DocumentType.ELECTRONIC, electronic_score
        elif scanned_score > 70 and electronic_score < 30:
            return DocumentType.SCANNED, scanned_score
        else:
            # 混合类型或不确定
            return DocumentType.MIXED, max(electronic_score, scanned_score) / 2


def get_optimal_parameters(doc_type):
    """
    根据文档类型获取最优参数配置

    Args:
        doc_type: DocumentType枚举值

    Returns:
        dict: 优化后的参数字典
    """
    if doc_type == DocumentType.ELECTRONIC:
        return {
            "color_tolerance": 18,  # 颜色严格匹配
            "inpaint_strength": 1.0,  # 中等修复强度
            "kernel_size": 3,  # 小核保持锐利
            "protect_text": True,  # 必须保护纯黑文字
            "multi_pass": 1,  # 单次足够
            "dpi": 150,  # 电子文档不需要高DPI
        }
    elif doc_type == DocumentType.SCANNED:
        return {
            "color_tolerance": 32,  # 宽松匹配应对色偏
            "inpaint_strength": 1.3,  # 更强修复力
            "kernel_size": 5,  # 大核处理模糊边缘
            "protect_text": True,  # 仍然保护文字
            "multi_pass": 2,  # 多轮处理残留
            "dpi": 200,  # 高DPI保留细节
        }
    else:  # MIXED
        return {
            "color_tolerance": 25,  # 平衡
            "inpaint_strength": 1.1,
            "kernel_size": 3,
            "protect_text": True,
            "multi_pass": 2,
            "dpi": 150,
        }
