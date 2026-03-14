"""
桌面宠物应用 - 重构版
功能：保持原有功能完全一致
优化：内存管理、日志清晰、代码结构化（Mixin模式拆分Murasame类）
"""

from PyQt5.QtMultimedia import QSound, QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QLabel, QSystemTrayIcon, QMenu, QAction, QGraphicsOpacityEffect
from PyQt5.QtGui import QPixmap, QIcon, QImage, QFont, QPainter, QFontDatabase, QColor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QRect, QSize, pyqtProperty, QUrl
from datetime import datetime
from Murasame import chat, generate, utils
import hashlib
import cv2
import threading
import textwrap
import os
import time
import sys
import json
import traceback
import pyautogui
import weakref
import logging
from typing import Optional, List, Dict, Any, Tuple


# 确保标准输出使用 UTF-8 编码，防止中文乱码
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# ============================================================================
# 日志管理模块
# ============================================================================

class PetLogger:
    """统一的日志管理器，提供清晰的日志输出格式"""

    # 日志级别颜色标记（终端）
    LEVEL_ICONS = {
        'DEBUG': '🔍',
        'INFO': 'ℹ️',
        'SUCCESS': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'PLAY': '🎵',
        'ANIM': '🎬',
        'INPUT': '⌨️',
        'LLM': '🤖',
    }

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.enabled = True
        self.debug_mode = False

    def _format_message(self, level: str, category: str, message: str) -> str:
        """格式化日志消息"""
        icon = self.LEVEL_ICONS.get(level, '•')
        timestamp = datetime.now().strftime('%H:%M:%S')
        if category:
            return f"[{timestamp}] {icon} [{category}] {message}"
        return f"[{timestamp}] {icon} {message}"

    def info(self, message: str, category: str = ''):
        """普通信息日志"""
        if self.enabled:
            print(self._format_message('INFO', category, message))

    def success(self, message: str, category: str = ''):
        """成功日志"""
        if self.enabled:
            print(self._format_message('SUCCESS', category, message))

    def warning(self, message: str, category: str = ''):
        """警告日志"""
        if self.enabled:
            print(self._format_message('WARNING', category, message))

    def error(self, message: str, category: str = ''):
        """错误日志"""
        if self.enabled:
            print(self._format_message('ERROR', category, message))

    def debug(self, message: str, category: str = ''):
        """调试日志（仅在debug模式下输出）"""
        if self.enabled and self.debug_mode:
            print(self._format_message('DEBUG', category, message))

    def play(self, message: str):
        """播放相关日志"""
        if self.enabled:
            print(self._format_message('PLAY', '播放', message))

    def anim(self, message: str):
        """动画相关日志"""
        if self.enabled:
            print(self._format_message('ANIM', '动画', message))

    def input(self, message: str):
        """输入相关日志"""
        if self.enabled:
            print(self._format_message('INPUT', '输入', message))

    def llm(self, message: str):
        """LLM相关日志"""
        if self.enabled:
            print(self._format_message('LLM', 'LLM', message))


# 全局日志实例
logger = PetLogger()


# ============================================================================
# 工具函数
# ============================================================================

def wrap_text(text: str) -> str:
    """文本换行处理"""
    config = utils.get_config()
    width = config.get('display', {}).get('text_width', 12)
    return '\n'.join(textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False))


def get_scale_factor() -> float:
    """获取HiDPI缩放因子"""
    scale_factor = 1.0
    config = utils.get_config()
    enable_auto = config.get('display', {}).get('scale', {}).get('enable_auto', True)
    if enable_auto:
        if hasattr(app, 'devicePixelRatio'):
            scale_factor = app.devicePixelRatio()
        elif hasattr(app.primaryScreen(), 'devicePixelRatio'):
            scale_factor = app.primaryScreen().devicePixelRatio()
    else:
        scale_factor = config.get('display', {}).get('scale', {}).get('custom_scale_factor', 1.0)

    return scale_factor


# ============================================================================
# 显示预设配置
# ============================================================================

class DisplayPresets:
    """显示预设配置管理"""

    PRESETS = {
        "compact": {
            "name": "紧凑模式",
            "visible_ratio": 0.35,
            "text_x_offset": 120,
            "text_y_offset": 15,
            "description": "最节省空间，只显示头部和肩部"
        },
        "balanced": {
            "name": "平衡模式",
            "visible_ratio": 0.45,
            "text_x_offset": 140,
            "text_y_offset": 20,
            "description": "推荐设置，显示上半身"
        },
        "standard": {
            "name": "标准模式",
            "visible_ratio": 0.6,
            "text_x_offset": 150,
            "text_y_offset": 25,
            "description": "显示到腰部，适中大小"
        },
        "full": {
            "name": "完整显示",
            "visible_ratio": 1.0,
            "text_x_offset": 160,
            "text_y_offset": -100,
            "description": "显示完整桌宠"
        }
    }

    @classmethod
    def get_preset(cls, preset_name: str) -> Tuple[bool, Dict]:
        """
        获取预设配置
        返回: (是否找到预设, 配置字典)
        """
        if preset_name in cls.PRESETS:
            return True, cls.PRESETS[preset_name]
        return False, cls.PRESETS['balanced']


# ============================================================================
# Murasame Mixin 类定义
# ============================================================================

class PlatformMixin:
    """平台特定功能Mixin - 处理macOS窗口层级等"""

    def _setup_macos_window_level(self):
        """在 macOS 上设置窗口层级，使其始终在最前但不抢占焦点"""
        import platform
        if platform.system() != 'Darwin':
            return

        try:
            from AppKit import NSApp, NSWindow, NSFloatingWindowLevel
            from PyQt5.QtGui import QWindow

            def set_level():
                try:
                    ns_view = self.winId()
                    if ns_view:
                        from objc import objc_object
                        import ctypes
                        ns_view_ptr = ctypes.c_void_p(int(ns_view))

                        from AppKit import NSView
                        view = objc_object(c_void_p=ns_view_ptr)
                        window = view.window()

                        if window:
                            window.setLevel_(NSFloatingWindowLevel)
                            logger.success("macOS window level set to NSFloatingWindowLevel", "平台")
                except Exception as e:
                    logger.error(f"Failed to set window level with PyObjC: {e}", "平台")

            QTimer.singleShot(100, set_level)
        except ImportError:
            logger.warning("PyObjC not available, falling back to Qt window flags", "平台")
        except Exception as e:
            logger.error(f"Cannot setup macOS window level: {e}", "平台")


class ImageProcessingMixin:
    """图像处理Mixin - 处理图像转换和缩放"""

    def cvimg_to_qpixmap(self, cv_img) -> QPixmap:
        """OpenCV图像转QPixmap"""
        if cv_img.shape[2] == 4:
            cv_img_bgra = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
            height, width, channel = cv_img_bgra.shape
            bytes_per_line = 4 * width
            qimg = QImage(cv_img_bgra.data, width, height,
                          bytes_per_line, QImage.Format_RGBA8888)
            return QPixmap.fromImage(qimg)
        return QPixmap()

    def _scale_pixmap(self, pixmap: QPixmap, scale_factor: float) -> QPixmap:
        """根据缩放因子缩放图像"""
        if scale_factor > 1.0:
            return pixmap.scaled(
                pixmap.width() // int(scale_factor * 2),
                pixmap.height() // int(scale_factor * 2),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        else:
            return pixmap.scaled(
                pixmap.width() // 2, pixmap.height() // 2,
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )


class AnimationMixin:
    """动画Mixin - 处理淡入淡出动画"""

    def _init_animation_state(self):
        """初始化动画状态"""
        self._xfade_old: Optional[QPixmap] = None
        self._xfade_new: Optional[QPixmap] = None
        self._xfade_t: float = 1.0
        self._xfade_anim = None

        # 淡入淡出背景标签
        self._fade_bg = QLabel(self)
        self._fade_fg = QLabel(self)
        for lbl in (self._fade_bg, self._fade_fg):
            lbl.setAttribute(Qt.WA_TranslucentBackground)
            lbl.setVisible(False)
            lbl.setGeometry(self.rect())
            lbl.lower()

        self._fade_bg_effect = QGraphicsOpacityEffect(self._fade_bg)
        self._fade_fg_effect = QGraphicsOpacityEffect(self._fade_fg)
        self._fade_bg.setGraphicsEffect(self._fade_bg_effect)
        self._fade_fg.setGraphicsEffect(self._fade_fg_effect)

    def _get_fade_progress(self) -> float:
        return self._xfade_t

    def _set_fade_progress(self, value: float):
        self._xfade_t = float(value)
        self.update()

    fadeProgress = pyqtProperty(float, fget=_get_fade_progress, fset=_set_fade_progress)

    def _cleanup_animation_resources(self):
        """清理动画资源"""
        self._xfade_old = None
        self._xfade_new = None
        if self._xfade_anim:
            self._xfade_anim.stop()
            self._xfade_anim = None


class TextDisplayMixin:
    """文本显示Mixin - 处理文本显示和打字效果"""

    def _init_text_state(self):
        """初始化文本显示状态"""
        self.display_text: str = ""
        self.text_font = QFont()
        self.text_font.setFamily("思源黑体 CN Bold")

        # 根据HiDPI调整字体大小
        scale_factor = get_scale_factor()
        font_size = 24
        if scale_factor > 1.0:
            font_size = int(24 / scale_factor)
        self.text_font.setPointSize(font_size)

        self.text_x_offset: int = 0
        self.text_y_offset: int = 0
        QFontDatabase.addApplicationFont("./思源黑体Bold.otf")

        # 打字效果
        self.full_text: str = ""
        self.typing_timer = QTimer()
        self.typing_timer.timeout.connect(self._typing_step)
        self.typing_interval: int = 40
        self._typing_index: int = 0
        self.typing_prefix: str = ""

        # 默认回复
        self.latest_response: str = "【 丛雨 】\n  「主人，你好呀！」"

    def show_text(self, text: str, x_offset: int = None, y_offset: int = None,
                  typing: bool = True, typing_prefix: str = "【 丛雨 】\n  "):
        """显示文本（支持打字效果）"""
        # 使用配置文件中的默认值
        if x_offset is None:
            x_offset = self.text_x_offset_default
        if y_offset is None:
            y_offset = self.text_y_offset_default

        # 根据缩放调整默认偏移量
        scale_factor = get_scale_factor()
        if scale_factor > 1.0:
            x_offset = int(x_offset / scale_factor)
            y_offset = int(y_offset / scale_factor)

        self.text_x_offset = x_offset
        self.text_y_offset = y_offset
        self.typing_prefix = typing_prefix

        if typing:
            if typing_prefix and text.startswith(typing_prefix):
                self.full_text = text[len(typing_prefix):]
                self.display_text = typing_prefix
            else:
                self.full_text = text
                self.display_text = ""
            self._typing_index = 0
            self.typing_timer.start(self.typing_interval)
        else:
            self.display_text = text
            self.full_text = text
            self.typing_timer.stop()
            self.update()

    def _typing_step(self):
        """打字效果单步"""
        if self._typing_index < len(self.full_text):
            self.display_text = self.typing_prefix + self.full_text[:self._typing_index + 1]
            self._typing_index += 1
            self.update()
        else:
            self.typing_timer.stop()

    def _draw_text_with_border(self, painter: QPainter, text_rect: QRect,
                               display_text: str, border_size: int):
        """绘制带边框的文本"""
        align_flag = Qt.AlignLeft | Qt.AlignBottom if '\n' in display_text else Qt.AlignHCenter | Qt.AlignBottom

        # 绘制边框
        painter.setPen(QColor(44, 22, 28))
        for dx, dy in [(-border_size, 0), (border_size, 0), (0, -border_size), (0, border_size),
                       (border_size, -border_size), (border_size, border_size),
                       (-border_size, -border_size), (-border_size, border_size)]:
            painter.drawText(text_rect.translated(dx, dy), align_flag, display_text)

        # 绘制文本
        painter.setPen(Qt.white)
        painter.drawText(text_rect, align_flag, display_text)


class InputMixin:
    """输入处理Mixin - 处理用户输入和输入法"""

    def _init_input_state(self):
        """初始化输入状态"""
        self.setAttribute(Qt.WA_InputMethodEnabled, True)
        self.input_mode: bool = False
        self.input_buffer: str = ""
        self.preedit_text: str = ""
        self.config = utils.get_config()
        self.user_name = self.config.get("user_name", "用户")

    def inputMethodQuery(self, query):
        """输入法查询"""
        if query == Qt.ImMicroFocus:
            rect = self.rect().adjusted(
                self.text_x_offset,
                self.text_y_offset,
                self.text_x_offset,
                -self.rect().height() // 2 + self.text_y_offset
            )
            pos = self.mapToGlobal(rect.bottomLeft())
            return QRect(pos, QSize(1, 30))
        return QLabel.inputMethodQuery(self, query)

    def inputMethodEvent(self, event):
        """输入法事件处理"""
        if self.input_mode:
            commit = event.commitString()
            preedit = event.preeditString()
            if commit:
                self.input_buffer += commit
            self.preedit_text = preedit
            wrapped = wrap_text(self.input_buffer + self.preedit_text)
            self.display_text = f"【 {self.user_name} 】\n  「{wrapped}」"
            self.update()
        else:
            QLabel.inputMethodEvent(self, event)

    def handle_user_input(self):
        """处理用户输入"""
        worker = globals().get('screen_worker')
        if worker and hasattr(worker, "interrupt_event"):
            worker.interrupt_event.set()

        self.llm_worker = LLMWorker(
            self.input_buffer, self.history, self.emotion_history,
            self.embeddings_history, role="user"
        )
        self.llm_worker.finished.connect(self.on_llm_result)
        self.llm_worker.start()

    def _handle_key_press(self, event):
        """处理按键事件"""
        if self.input_mode:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.input_mode = False
                self.handle_user_input()
            elif event.key() == Qt.Key_Backspace:
                if self.preedit_text:
                    pass
                else:
                    self.input_buffer = self.input_buffer[:-1]
                    wrapped = wrap_text(self.input_buffer)
                    if not wrapped.strip():
                        self.display_text = f"【 {self.user_name} 】\n  ..."
                    else:
                        self.display_text = f"【 {self.user_name} 】\n  「{wrapped}」"
                    self.update()
            else:
                char = event.text()
                if char and not self.preedit_text:
                    self.input_buffer += char
                    wrapped = wrap_text(self.input_buffer)
                    if not wrapped.strip():
                        self.display_text = f"【 {self.user_name} 】\n  ..."
                    else:
                        self.display_text = f"【 {self.user_name} 】\n  「{wrapped}」"
                    self.update()
        else:
            QLabel.keyPressEvent(self, event)


class InteractionMixin:
    """交互处理Mixin - 处理鼠标事件"""

    def _init_interaction_state(self):
        """初始化交互状态"""
        self.mousePressEvent = self.start_move
        self.mouseMoveEvent = self.on_move
        self.offset = None
        self.touch_head: bool = False
        self.head_press_x = None
        self.config = utils.get_config()
        self.user_name = self.config.get("user_name", "用户")

    def start_move(self, event):
        """开始移动/交互"""
        if event.button() == Qt.LeftButton:
            rect = self.rect()

            # 计算头部区域
            visible_height = int(self.height() * self.visible_ratio)
            head_threshold = visible_height // 2

            if event.y() < head_threshold:
                self.touch_head = True
                self.head_press_x = event.x()
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.touch_head = False
                self.head_press_x = None
                self.setCursor(Qt.ArrowCursor)

            # 检查是否点击了文本区域
            text_clicked = False
            if self.display_text:
                rect = self.rect()
                text_rect = rect.adjusted(
                    self.text_x_offset,
                    self.text_y_offset,
                    self.text_x_offset,
                    -rect.height() // 2 + self.text_y_offset
                )
                expanded_rect = text_rect.adjusted(-20, -20, 20, 20)
                if expanded_rect.contains(event.pos()):
                    text_clicked = True

            # 输入区域判断
            input_threshold = int(visible_height * 0.7)
            if event.y() > input_threshold or text_clicked:
                self.input_mode = True
                self.input_buffer = ""
                self.display_text = f"【 {self.user_name} 】\n  ..."
                self.update()
                return

        if event.button() == Qt.MiddleButton:
            self.offset = event.pos()
            self.setCursor(Qt.SizeAllCursor)

    def on_move(self, event):
        """移动事件处理"""
        # 摸头交互
        if self.touch_head and self.head_press_x is not None and event.buttons() & Qt.LeftButton:
            if abs(event.x() - self.head_press_x) > 50:
                self.llm_worker = LLMWorker(
                    "主人摸了摸你的头", self.history, self.emotion_history,
                    self.embeddings_history, role="system"
                )
                self.llm_worker.finished.connect(self.on_llm_result)
                self.llm_worker.start()
                self.touch_head = False
                self.head_press_x = None

        # 中键拖动窗口
        if self.offset is not None and event.buttons() == Qt.MiddleButton:
            self.move(self.pos() + event.pos() - self.offset)

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MiddleButton:
            self.offset = None
            self.setCursor(Qt.ArrowCursor)
        if event.button() == Qt.LeftButton:
            self.touch_head = False
            self.head_press_x = None
            self.setCursor(Qt.ArrowCursor)


class PlaybackMixin:
    """播放队列Mixin - 处理语音播放队列"""

    def _init_playback_state(self):
        """初始化播放状态"""
        # 播放队列系统
        self._play_queue: List[Dict] = []
        self._is_playing: bool = False
        self._current_playback: Optional[Dict] = None
        self._current_sentence_index: int = 0
        self._sentence_queue: List[Dict] = []

        # 音频播放器
        self._media_player = QMediaPlayer()
        self._media_player.stateChanged.connect(self._on_media_state_changed)
        self._pending_play = None

    def _on_media_state_changed(self, state):
        """音频播放完毕后的回调"""
        if state != QMediaPlayer.StoppedState or not self._pending_play:
            return

        self._pending_play = None
        self._current_sentence_index += 1

        if self._current_sentence_index < len(self._sentence_queue):
            QTimer.singleShot(100, self._play_next_sentence)
        else:
            self._move_to_next_request()

    def _move_to_next_request(self):
        """切换到下一个播放请求"""
        if self._play_queue:
            logger.play(f"等待500ms后开始下一个请求 (队列中还有 {len(self._play_queue)} 个请求)")
            QTimer.singleShot(500, self._start_playback)
        else:
            self._is_playing = False
            self._current_playback = None
            logger.success("所有播放请求完成", "播放")

    def _start_playback(self):
        """开始播放队列中的第一个请求"""
        if not self._play_queue:
            return

        self._is_playing = True
        self._current_playback = self._play_queue.pop(0)
        self._sentence_queue = self._current_playback['sentences_data'].copy()
        self._current_sentence_index = 0

        logger.play(f"开始播放队列中的请求: {len(self._sentence_queue)} 个句子")
        self._play_next_sentence()

    def _play_next_sentence(self):
        """播放下一个句子"""
        if not hasattr(self, '_sentence_queue') or self._current_sentence_index >= len(self._sentence_queue):
            logger.success("当前请求的所有句子播放完毕", "播放")
            self._move_to_next_request()
            return

        data = self._sentence_queue[self._current_sentence_index]
        logger.play(f"播放句子 {self._current_sentence_index + 1}/{len(self._sentence_queue)}: {data['original']}")

        # 切换立绘
        self.switch_image(data['pose'], data['embeddings_layers'])

        # 更新最新回复
        self.latest_response = f"【 丛雨 】\n  「{wrap_text(data['original'])}」"

        # 处理错误类型
        if data.get('is_error'):
            prefix = data.get('error_title', '【 系统错误 】')
            self.show_text(f"{prefix}\n  {data['original']}", typing=False)
            self._current_sentence_index += 1
            QTimer.singleShot(100, self._play_next_sentence)
            return

        # 正常显示和播放
        self.show_text(self.latest_response, typing=True)

        voice_path = os.path.join(os.getcwd(), 'voices', f"{data['voice_md5']}.wav")
        self._pending_play = (voice_path, data)
        self._media_player.setMedia(QMediaContent(QUrl.fromLocalFile(voice_path)))
        self._media_player.play()
        logger.play(f"开始播放音频: {data['voice_md5']}")


# ============================================================================
# 主类 Murasame - 组合所有Mixin
# ============================================================================

class Murasame(QLabel,
               PlatformMixin,
               ImageProcessingMixin,
               AnimationMixin,
               TextDisplayMixin,
               InputMixin,
               InteractionMixin,
               PlaybackMixin):
    """桌面宠物主类 - 组合所有功能模块"""

    # 显示预设配置（保持原有接口）
    DISPLAY_PRESETS = DisplayPresets.PRESETS

    def __init__(self):
        super().__init__()

        # ========== 初始化显示配置 ==========
        self._init_display_config()

        # ========== 初始化历史记录 ==========
        self.history = chat.identity()
        self.emotion_history: List = []
        self.embeddings_history: List = []

        # ========== 初始化各模块状态 ==========
        self._init_animation_state()
        self._init_text_state()
        self._init_input_state()
        self._init_interaction_state()
        self._init_playback_state()

        # ========== 设置窗口属性 ==========
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._setup_macos_window_level()

        # ========== 加载初始图像 ==========
        self._load_initial_image()

    def _init_display_config(self):
        """初始化显示配置"""
        config = utils.get_config()
        display_config = config.get('display', {})
        preset_name = display_config.get('preset', 'balanced')

        if preset_name in self.DISPLAY_PRESETS:
            preset = self.DISPLAY_PRESETS[preset_name]
            self.visible_ratio = preset['visible_ratio']
            self.text_x_offset_default = preset['text_x_offset']
            self.text_y_offset_default = preset['text_y_offset']
            logger.success(f"使用显示预设: {preset['name']} - {preset['description']}", "显示")
        elif preset_name == 'custom':
            custom_config = display_config.get('custom', {})
            self.visible_ratio = custom_config.get('visible_ratio', 0.4)
            self.text_x_offset_default = custom_config.get('text_x_offset', 140)
            self.text_y_offset_default = custom_config.get('text_y_offset', 20)
            logger.success("使用自定义显示配置", "显示")
        else:
            preset = self.DISPLAY_PRESETS['balanced']
            self.visible_ratio = preset['visible_ratio']
            self.text_x_offset_default = preset['text_x_offset']
            self.text_y_offset_default = preset['text_y_offset']
            logger.warning(f"未知预设 '{preset_name}'，使用默认: {preset['name']}", "显示")

    def _load_initial_image(self):
        """加载初始图像"""
        cv_img = generate.generate_fgimage(target="ムラサメb", embeddings_layers=[1716, 1475, 1261])
        pixmap = self.cvimg_to_qpixmap(cv_img)

        scale_factor = get_scale_factor()
        pixmap = self._scale_pixmap(pixmap, scale_factor)

        self.setPixmap(pixmap)
        self.resize(pixmap.size())

    def event(self, event):
        """事件处理"""
        worker = globals().get('screen_worker')

        if event.type() == QEvent.WindowActivate:
            logger.debug("窗口激活", "事件")
            if worker:
                worker.should_capture = False
                if hasattr(worker, "interrupt_event"):
                    worker.interrupt_event.set()
        elif event.type() == QEvent.WindowDeactivate:
            logger.debug("窗口失活", "事件")
            self.input_mode = False
            self.show_text(self.latest_response, typing=True)
            if worker:
                worker.should_capture = True
        return QLabel.event(self, event)

    def paintEvent(self, event):
        """绘制事件"""
        if self._xfade_old is not None and self._xfade_new is not None:
            self._paint_crossfade()
        else:
            QLabel.paintEvent(self, event)
            if self.display_text:
                self._paint_text()

    def _paint_crossfade(self):
        """绘制淡入淡出效果"""
        w, h = self.width(), self.height()

        # 创建旧图像
        img_old = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img_old.fill(0)
        p = QPainter(img_old)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawPixmap(0, 0, self._xfade_old)
        p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        p.fillRect(img_old.rect(), QColor(0, 0, 0, int((1.0 - self._xfade_t) * 255)))
        p.end()

        # 创建新图像
        img_new = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        img_new.fill(0)
        p = QPainter(img_new)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawPixmap(0, 0, self._xfade_new)
        p.setCompositionMode(QPainter.CompositionMode_DestinationIn)
        p.fillRect(img_new.rect(), QColor(0, 0, 0, int(self._xfade_t * 255)))
        p.end()

        # 混合图像
        blended = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
        blended.fill(0)
        p = QPainter(blended)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.drawImage(0, 0, img_old)
        p.setCompositionMode(QPainter.CompositionMode_Plus)
        p.drawImage(0, 0, img_new)
        p.end()

        # 绘制到窗口
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawImage(0, 0, blended)

        if self.display_text:
            self._draw_text(painter)

        painter.end()

    def _paint_text(self):
        """绘制文本"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        self._draw_text(painter)
        painter.end()

    def _draw_text(self, painter: QPainter):
        """绘制文本（内部方法）"""
        painter.setFont(self.text_font)
        rect = self.rect()
        text_rect = rect.adjusted(
            self.text_x_offset,
            self.text_y_offset,
            self.text_x_offset,
            -rect.height() // 2 + self.text_y_offset
        )

        scale_factor = get_scale_factor()
        border_size = max(1, int(2 / scale_factor))
        self._draw_text_with_border(painter, text_rect, self.display_text, border_size)

    def keyPressEvent(self, event):
        """按键事件"""
        self._handle_key_press(event)

    def switch_image(self, target: str, embeddings_layers: List):
        """切换立绘图像"""
        cv_img = generate.generate_fgimage(
            target=f"ムラサメ{target}",
            embeddings_layers=embeddings_layers
        )
        pixmap_new = self.cvimg_to_qpixmap(cv_img)

        scale_factor = get_scale_factor()
        pixmap_new = self._scale_pixmap(pixmap_new, scale_factor)

        pixmap_old = self.pixmap()
        if pixmap_old is None:
            self.setPixmap(pixmap_new)
            self.resize(pixmap_new.size())
            self.update()
            return

        self._xfade_old = pixmap_old
        self._xfade_new = pixmap_new
        self._xfade_t = 0.0

        if self._xfade_anim:
            self._xfade_anim.stop()

        from PyQt5.QtCore import QPropertyAnimation
        self._xfade_anim = QPropertyAnimation(self, b"fadeProgress")
        self._xfade_anim.setDuration(400)
        self._xfade_anim.setStartValue(0.0)
        self._xfade_anim.setEndValue(1.0)

        def finish():
            self.setPixmap(pixmap_new)
            self.resize(pixmap_new.size())
            self._cleanup_animation_resources()
            self.update()

        self._xfade_anim.finished.connect(finish)
        self._xfade_anim.start()
        logger.anim(f"切换立绘: {target}")

    def on_llm_result(self, sentences_data, history, emotion_history, embeddings_history):
        """处理LLM结果"""
        # 检查是否是错误信号（旧版兼容性）
        if isinstance(sentences_data, str):
            self.show_text(sentences_data, typing=False)
            return

        if not sentences_data or not isinstance(sentences_data, list):
            logger.warning("没有句子数据", "LLM")
            return

        # 保存历史记录
        self.history = history
        self.emotion_history = emotion_history
        self.embeddings_history = embeddings_history

        # 清空输入缓冲区
        self.input_buffer = ""
        self.preedit_text = ""

        # 将播放请求添加到队列
        play_request = {
            'sentences_data': sentences_data.copy(),
            'history': history,
            'emotion_history': emotion_history,
            'embeddings_history': embeddings_history
        }

        self._play_queue.append(play_request)
        logger.llm(f"添加到播放队列: {len(sentences_data)} 个句子 (队列长度: {len(self._play_queue)})")

        # 如果当前没有在播放，开始播放
        if not self._is_playing:
            self._start_playback()

    def cleanup(self):
        """清理资源"""
        self.typing_timer.stop()
        self._cleanup_animation_resources()
        self._media_player.stop()
        logger.info("资源清理完成", "系统")


# ============================================================================
# ScreenWorker - 屏幕监控线程
# ============================================================================

class ScreenWorker(QThread):
    """屏幕监控工作线程"""
    screen_result = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running: bool = True
        self.history: List = []
        self.should_capture: bool = False
        self.llmworker: Optional['LLMWorker'] = None
        self.interrupt_event = threading.Event()

    def run(self):
        """线程主循环"""
        while self.running:
            logger.debug(f"should_capture: {self.should_capture}", "屏幕监控")

            if self.should_capture:
                self.interrupt_event.clear()
                try:
                    screenshot = pyautogui.screenshot()
                    sys_prompt = '''你现在要担任一个AI桌宠的视觉识别助手，我会向你提供用户此时的屏幕截图，你要识别用户此时的行为，并进行描述。我会将你的描述以system消息提供给另外一个处理语言的AI模型。'''

                    try:
                        response, _ = chat.query_image(
                            screenshot,
                            "现在请描述用户此时的行为",
                            [{"role": "system", "content": sys_prompt}]
                        )

                        if not response:
                            continue

                        des, self.history = chat.think_image(response, self.history)
                        if des and des.get('des'):
                            logger.info(f"屏幕识别: {des['des']}", "视觉")
                            self.screen_result.emit(des['des'])

                    except Exception as inner_e:
                        logger.error(f"视觉识别环节出错 (网络/API问题): {inner_e}", "视觉")

                except Exception as e:
                    logger.error(f"ScreenWorker 关键错误: {e}", "屏幕监控")

            config = utils.get_config()
            screenworker_delay = config.get("screenworker_delay", 30)
            time.sleep(screenworker_delay)


# ============================================================================
# LLMWorker - LLM处理线程
# ============================================================================

class LLMWorker(QThread):
    """LLM处理工作线程"""
    finished = pyqtSignal(list, list, list, list)  # sentences_data, history, emotion_history, embeddings_history

    def __init__(self, prompt: str, history: List, emotion_history: List,
                 embeddings_history: List, role: str = "user", interrupt_event=None):
        super().__init__()
        self.prompt = prompt
        self.history = history
        self.role = role
        self.emotion_history = emotion_history
        self.embeddings_history = embeddings_history
        self.interrupt_event = interrupt_event

    def _get_time_period(self) -> Tuple[str, int, int]:
        """获取当前时间段"""
        hour = datetime.now().hour
        minute = datetime.now().minute

        if 0 <= hour < 5:
            period = "凌晨"
        elif 5 <= hour < 12:
            period = "早上"
        elif 12 <= hour < 18:
            period = "下午"
        else:
            period = "晚上"

        return period, hour, minute

    def _check_interrupted(self) -> bool:
        """检查是否被中断"""
        if self.interrupt_event and self.interrupt_event.is_set():
            logger.info("LLMWorker interrupted", "LLM")
            return True
        return False

    def _process_sentence(self, indexed_item: Tuple[int, Tuple[str, str]],
                          emotions_list: List[str], costume: str) -> Optional[Dict]:
        """处理单个句子"""
        if self._check_interrupted():
            return None

        idx, (sent, pose) = indexed_item
        curr_emotion = emotions_list[idx] if idx < len(emotions_list) else "平静"

        # 翻译
        translated = chat.get_translate(sent)
        if self._check_interrupted():
            return None

        # 生成TTS
        voice_md5 = chat.generate_tts(translated, curr_emotion)
        if self._check_interrupted():
            return None

        # 获取立绘表情层
        embeddings_layers, _ = chat.get_embedings_layers(sent, pose, costume, [])
        if self._check_interrupted():
            return None

        return {
            'idx': idx,
            'original': sent,
            'translated': translated,
            'pose': pose,
            'emotion': curr_emotion,
            'voice_md5': voice_md5,
            'embeddings_layers': embeddings_layers
        }

    def _write_error_log(self, error: Exception):
        """写入错误日志"""
        error_log_dir = os.path.join('logs', 'error')
        os.makedirs(error_log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        error_log_path = os.path.join(error_log_dir, f'error_{timestamp}.log')

        tb_str = traceback.format_exc()
        log_content = f"""--- LLMWorker Error Log ---
Timestamp: {datetime.now().isoformat()}

Prompt that caused the error:
------------------------------
{self.prompt}

Full Conversation History (at time of error):
---------------------------------------------
{json.dumps(self.history, indent=2, ensure_ascii=False)}

Traceback:
----------
{tb_str}
"""
        try:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            logger.success(f"错误日志已保存: {error_log_path}", "错误")
        except Exception as log_e:
            logger.error(f"写入错误日志失败: {log_e}", "错误")

    def run(self):
        """线程主逻辑"""
        try:
            t_start = time.time()

            # 添加时间信息
            period, hour, minute = self._get_time_period()
            self.history.append({"role": "system", "content": f"现在是{period}{hour}点{minute}分"})

            if self._check_interrupted():
                return

            # 查询LLM
            response, history = chat.query(
                prompt=self.prompt,
                history=self.history,
                role=self.role
            )

            if self._check_interrupted():
                return

            # 分割句子
            sentence_splits, costume = chat.split_sentence(response, [])
            logger.llm(f"句子分割结果: {len(sentence_splits)} 个句子，统一服装: {costume}")
            for i, (sent, pose) in enumerate(sentence_splits):
                logger.debug(f"  [{i + 1}] ({pose}) {sent}")

            # 获取情绪列表
            sentences_only = [s[0] for s in sentence_splits]
            emotions_list, _ = chat.get_emotion(response, sentences_only, [])
            logger.llm(f"情绪列表: {emotions_list}")

            if self._check_interrupted():
                return

            # 并行处理所有句子
            import concurrent.futures
            sentences_data = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self._process_sentence, item, emotions_list, costume)
                    for item in enumerate(sentence_splits)
                ]
                for future in concurrent.futures.as_completed(futures):
                    if self._check_interrupted():
                        return
                    result = future.result()
                    if result:
                        sentences_data.append(result)

            # 按原始索引排序
            sentences_data.sort(key=lambda x: x['idx'])

            if self._check_interrupted():
                return

            # 等待所有语音文件生成完成
            logger.info("等待所有语音文件生成...", "LLM")
            for data in sentences_data:
                voice_path = f"./voices/{data['voice_md5']}.wav"
                while not os.path.exists(voice_path):
                    time.sleep(0.1)
                    if self._check_interrupted():
                        return


            elapsed = time.time() - t_start
            logger.success(f"所有 {len(sentences_data)} 个句子处理完成，耗时 {elapsed:.2f}s", "LLM")

            self.finished.emit(sentences_data, history, self.emotion_history, self.embeddings_history)

        except Exception as e:
            logger.error(f"LLMWorker 错误: {e}", "LLM")
            self._write_error_log(e)

            # 发送错误信号
            error_data = [{
                'idx': 0,
                'original': type(e).__name__,
                'translated': '',
                'pose': 'b',
                'emotion': '平静',
                'voice_md5': '',
                'embeddings_layers': [1716, 1475, 1261],
                'is_error': True,
                'error_title': "【 系统错误 】"
            }]
            self.finished.emit(error_data, self.history, self.emotion_history, self.embeddings_history)


# ============================================================================
# 工具函数
# ============================================================================

def clear_history(parent):
    """清除历史记录"""
    from PyQt5.QtWidgets import QMessageBox
    reply = QMessageBox.question(
        parent, "Clear History",
        "Are you sure you want to clear the history?",
        QMessageBox.Ok | QMessageBox.Cancel
    )
    if reply == QMessageBox.Ok:
        murasame.history = chat.identity()
        murasame.emotion_history = []
        murasame.embeddings_history = []
        logger.success("历史记录已清除", "系统")


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    history = chat.identity()

    app = QApplication(sys.argv)
    murasame = Murasame()

    # 动态计算窗口位置
    screen = app.primaryScreen()
    screen_geometry = screen.availableGeometry()
    window_width = murasame.width()
    window_height = murasame.height()

    # 放在右下角，只显示上半身
    x = screen_geometry.width() - window_width - 20
    y = screen_geometry.height() - int(window_height * murasame.visible_ratio)

    murasame.move(x, y)
    murasame.show()

    # 系统托盘
    murasame.tray_icon = QSystemTrayIcon(QIcon("icon.png"), parent=app)
    tray_menu = QMenu()

    clear_action = QAction("Clear History")
    clear_action.triggered.connect(lambda: clear_history(murasame))
    exit_action = QAction("Exit")
    exit_action.triggered.connect(app.quit)

    tray_menu.addAction(clear_action)
    tray_menu.addAction(exit_action)

    murasame.tray_icon.setContextMenu(tray_menu)
    murasame.tray_icon.show()

    # 显示初始文本
    murasame.show_text(murasame.latest_response, typing=True)

    # 屏幕监控
    screen_worker = None

    if utils.get_config()['enable_vl']:
        screen_worker = ScreenWorker()


        def handle_screen_result(des_text):
            murasame.llm_worker = LLMWorker(
                des_text, murasame.history, murasame.emotion_history,
                murasame.embeddings_history, role="system"
            )
            murasame.llm_worker.finished.connect(murasame.on_llm_result)
            murasame.llm_worker.start()


        screen_worker.screen_result.connect(handle_screen_result)
        screen_worker.start()

    sys.exit(app.exec_())
