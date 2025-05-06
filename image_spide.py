import os
import requests
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QGridLayout, QMessageBox, QProgressBar
)
from PyQt5.QtGui import QFont, QIcon, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 将 API 密钥写入代码
API_KEY = "47404800-1b72feca4ed91c4712fd11988"


def download_image(url, save_dir, filename):
    """下载图片并保存到本地"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        file_path = os.path.join(save_dir, filename)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    except Exception as e:
        print(f"下载失败：{url}, 错误信息：{e}")


def fetch_images_from_api(keyword, num_images, progress_callback):
    """通过 Pixabay API 获取图片并保存"""
    api_url = "https://pixabay.com/api/"
    params = {
        "key": API_KEY,
        "q": keyword,
        "image_type": "photo",
        "per_page": num_images,
        "safesearch": "true"
    }

    save_dir = os.path.join("./images", keyword)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    try:
        # 发起 API 请求
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "hits" not in data or len(data["hits"]) == 0:
            return None  # 表示没有找到图片

        # 下载图片
        for idx, hit in enumerate(data["hits"], 1):
            img_url = hit.get("largeImageURL")
            if img_url:
                filename = f"{keyword}_{idx}.jpg"
                download_image(img_url, save_dir, filename)
                progress_callback.emit(idx, num_images)  # 更新进度条

        return True
    except Exception as e:
        print(f"调用 Pixabay API 出现错误：{e}")
        return False

class DownloadThread(QThread):
    """线程类，用于处理下载过程中的任务"""
    progress_signal = pyqtSignal(int, int)  # 进度条更新信号

    def __init__(self, keyword, num_images):
        super().__init__()
        self.keyword = keyword
        self.num_images = num_images

    def run(self):
        fetch_images_from_api(self.keyword, self.num_images, self.progress_signal)


class ImageDownloaderApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Shalee的图片下载工具")
        self.setWindowIcon(QIcon("icon.png"))  # 替换为高清图标路径
        self.resize(600, 400)

        # 主布局
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # 标题部分
        title_label = QLabel("Shalee的图片下载工具")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Logo 部分
        logo = QLabel()
        logo.setPixmap(QPixmap("logo.png").scaled(100, 100, Qt.KeepAspectRatio))  # 替换为合适的 logo
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)

        # 输入区域
        input_layout = QGridLayout()
        input_layout.setSpacing(10)

        # 关键词输入框
        keyword_label = QLabel("关键词：")
        keyword_label.setFont(QFont("Arial", 16))
        self.keyword_input = QLineEdit(self)
        self.keyword_input.setPlaceholderText("请输入关键词（例如：cat）")
        self.keyword_input.setFont(QFont("Arial", 16))

        # 图片数量输入框
        num_label = QLabel("图片数量：")
        num_label.setFont(QFont("Arial", 16))
        self.num_input = QLineEdit(self)
        self.num_input.setPlaceholderText("请输入图片数量")
        self.num_input.setFont(QFont("Arial", 16))

        # 布局控件
        input_layout.addWidget(keyword_label, 0, 0)
        input_layout.addWidget(self.keyword_input, 0, 1)
        input_layout.addWidget(num_label, 1, 0)
        input_layout.addWidget(self.num_input, 1, 1)

        layout.addLayout(input_layout)

        # 下载按钮
        download_button = QPushButton("下载图片", self)
        download_button.setFont(QFont("Arial", 18, QFont.Bold))
        download_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
        """)
        download_button.clicked.connect(self.start_download)
        layout.addWidget(download_button, alignment=Qt.AlignCenter)

        # 提示信息
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFont(QFont("Arial", 14))
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 设置主布局
        self.setLayout(layout)

    def start_download(self):
        keyword = self.keyword_input.text().strip()
        num_images_text = self.num_input.text().strip()

        # 验证图片数量输入
        try:
            num_images = int(num_images_text)
            if num_images <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "错误", "请输入有效的图片数量！")
            return

        if not keyword:
            QMessageBox.warning(self, "错误", "关键词不能为空！")
            return

        self.setEnabled(False)
        self.status_label.setText("正在下载图片，请稍候...")

        self.thread = DownloadThread(keyword, num_images)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished.connect(self.on_download_finished)
        self.thread.start()

    def update_progress(self, current, total):
        """更新进度条"""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)

    def on_download_finished(self):
        self.setEnabled(True)
        self.status_label.setText("下载完成！")

        QMessageBox.information(self, "成功", "所有图片下载成功！")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # 现代界面风格
    app.setStyleSheet("""
        QWidget {
            background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #FFC0CB, stop:1 #FFD700);
        }
        QLabel {
            color: #333;
        }
        QLineEdit, QComboBox {
            background-color: white;
            border: 1px solid #CCC;
            border-radius: 5px;
            padding: 5px;
        }
        QComboBox QAbstractItemView {
            background-color: white;
        }
    """)

    window = ImageDownloaderApp()
    window.show()
    sys.exit(app.exec_())