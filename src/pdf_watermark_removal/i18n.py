"""Internationalization support for the watermark removal tool."""

import locale

# Supported languages
LOCALES = {
    "zh_CN": {
        "title": "PDF水印移除工具",
        "watermark_detection": "水印颜色检测",
        "recommended_color": "推荐的水印颜色",
        "confidence": "置信度",
        "coverage": "覆盖率",
        "rgb_value": "RGB值",
        "gray_level": "灰度级别",
        "use_color": "使用此颜色?",
        "show_alternatives": "显示其他选项?",
        "select_color": "选择颜色编号",
        "other_colors": "其他检测的颜色",
        "processing_pages": "处理页面",
        "step_1": "第1步: 将PDF转换为图像",
        "step_2": "第2步: 移除水印",
        "step_3": "第3步: 将图像转换回PDF",
        "loading_pdf": "加载PDF",
        "removing_watermarks": "移除水印",
        "saving_pdf": "保存PDF",
        "success": "成功",
        "completed": "水印移除完成!",
        "output_saved": "输出已保存",
        "pages_processed": "处理的页数",
        "pixels_removed": "移除的像素",
        "time_elapsed": "用时",
        "using_auto_detection": "使用自动检测",
        "using_recommended": "使用推荐颜色",
        "select_number": "选择颜色编号（或'a'使用自动）",
        "invalid_choice": "无效选择",
        "auto_detection": "自动检测",
    },
    "en_US": {
        "title": "PDF Watermark Removal Tool",
        "watermark_detection": "WATERMARK COLOR DETECTION",
        "recommended_color": "Recommended Watermark Color",
        "confidence": "Confidence",
        "coverage": "Coverage",
        "rgb_value": "RGB Value",
        "gray_level": "Gray Level",
        "use_color": "Use this color",
        "show_alternatives": "Show alternative candidates",
        "select_color": "Select color number",
        "other_colors": "Other detected colors",
        "processing_pages": "Processing pages",
        "step_1": "Step 1: Converting PDF to images",
        "step_2": "Step 2: Removing watermarks",
        "step_3": "Step 3: Converting images back to PDF",
        "loading_pdf": "Loading PDF",
        "removing_watermarks": "Removing watermarks",
        "saving_pdf": "Saving PDF",
        "success": "Success",
        "completed": "Watermark removal completed successfully!",
        "output_saved": "Output saved to",
        "pages_processed": "Pages processed",
        "pixels_removed": "Pixels removed",
        "time_elapsed": "Time elapsed",
        "using_auto_detection": "Using automatic detection",
        "using_recommended": "Using recommended color",
        "select_number": "Select color number (or 'a' for auto)",
        "invalid_choice": "Invalid choice",
        "auto_detection": "Automatic detection",
    },
}


def get_system_locale():
    """Detect system locale.

    Returns:
        str: Language code (e.g., 'zh_CN', 'en_US')
    """
    try:
        system_locale = locale.getdefaultlocale()[0]
        if system_locale and system_locale in LOCALES:
            return system_locale
        # Check if language code matches
        for loc in LOCALES:
            if loc.startswith(system_locale.split("_")[0]):
                return loc
    except Exception:
        pass

    # Default to English
    return "en_US"


class Translator:
    """Translation helper."""

    def __init__(self, language=None):
        """Initialize translator.

        Args:
            language: Language code (e.g., 'zh_CN'). Uses system locale if None.
        """
        if language is None:
            language = get_system_locale()

        self.language = language if language in LOCALES else "en_US"
        self.messages = LOCALES[self.language]

    def t(self, key, **kwargs):
        """Translate a message.

        Args:
            key: Message key
            **kwargs: Format arguments

        Returns:
            str: Translated message
        """
        message = self.messages.get(key, key)
        if kwargs:
            try:
                return message.format(**kwargs)
            except Exception:
                return message
        return message


# Global translator instance
_translator = None


def set_language(language):
    """Set the global language.

    Args:
        language: Language code (e.g., 'zh_CN')
    """
    global _translator
    _translator = Translator(language)


def get_translator():
    """Get the global translator instance.

    Returns:
        Translator: Global translator
    """
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator


def t(key, **kwargs):
    """Translate a message using the global translator.

    Args:
        key: Message key
        **kwargs: Format arguments

    Returns:
        str: Translated message
    """
    return get_translator().t(key, **kwargs)
