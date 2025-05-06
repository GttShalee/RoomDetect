import os
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
    QWidget, QStackedWidget, QScrollArea, QGridLayout, QMessageBox, QProgressBar, QLineEdit, QListWidget,
    QListWidgetItem
)
from matplotlib.ticker import MaxNLocator

from image_spide import DownloadThread
from yolov5_detector import YOLOv5Detector  # 导入yolo检测类

from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QSize
import matplotlib.pyplot as plt

# 将 API 密钥写入代码  这是Pixabay网站上的，但是这个网站基本上都是风景图而且是国外的，库存比较少
# 之前用 baidu 做了一个但是百度那个爬的质量太低了，100个里面能出10个hhh🥺🥺🥺 回来试试用bing引擎
API_KEY = "47404800-1b72feca4ed91c4712fd11988"

# 配置模型路径  训练的数据集特别少哈哈哈哈，几十张照片用我的cpu训了1个小时干冒烟了快
MODEL_PATH = "./yolov5/runs/train/exp/weights/best.pt"

class MainWindow(QMainWindow):
    """
    这是主窗口
    其实本来没想着做这么多，但是后面弄着弄着就多了，一开始不咋会用python的类和对象，就全部一个文件了，然后就依托了
    """

    def set_background_image(self):
        # 设置背景图片路径
        background_image_path = "/Users/leeshal/Desktop/Yolo_V5/background1.jpg"  # 替换为你的图片路径
        self.setStyleSheet(f"""
            QMainWindow {{
                background-image: url({background_image_path});
                background-repeat: no-repeat;
                background-position: center;
                background-attachment: fixed;
            }}
        """)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Shalee的多功能图像检测系统")
        self.setGeometry(200, 200, 1200, 800)
        # self.set_background_image()

        # 初始化检测器
        self.detector = YOLOv5Detector(model_path=MODEL_PATH, device="cpu")

        # 顶部栏
        self.header_label = QLabel("监测")
        self.header_label.setAlignment(Qt.AlignCenter)
        self.header_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                background-color: #2196F3;
                color: white;
                padding: 10px;
            }
        """)

        # 侧边栏和页面布局
        self.initUI()

    def initUI(self):
        # 创建侧边栏
        self.sidebar = QVBoxLayout()
        self.sidebar.setSpacing(20)
        self.sidebar.setContentsMargins(20, 20, 20, 20)

        # 创建页面容器
        self.pages = QStackedWidget()

        # 初始化页面内容
        self.create_sidebar_buttons()
        self.create_pages()

        # 布局设置
        main_layout = QHBoxLayout()
        main_layout.addLayout(self.sidebar, 1)
        main_layout.addWidget(self.pages, 4)

        # 顶部栏和主布局组合
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(self.header_label, 1)
        container_layout.addLayout(main_layout, 10)

        self.setCentralWidget(container)

    def switch_page(self, page_name):
        # 页面切换逻辑
        self.header_label.setText(page_name)
        page_map = {
            "监测": self.page_monitor,
            "用户管理": self.page_user,
            "图片下载": self.page_download,
            "数学函数图形化": self.page_math,
            "结果保存": self.page_result,
        }
        self.pages.setCurrentWidget(page_map[page_name])

    def create_sidebar_buttons(self):
        # 创建侧边栏按钮
        self.buttons = {
            "监测": QPushButton("监测"),
            "用户管理": QPushButton("用户管理"),
            "图片下载": QPushButton("图片下载"),
            "数学函数图形化": QPushButton("数学函数图形化"),
            "结果保存": QPushButton("结果保存"),
        }

        for btn_text, btn in self.buttons.items():
            # 设置按钮样式
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;  /* 绿色背景 */
                    color: white;  /* 白色文字 */
                    border: none;
                    border-radius: 12px;  /* 圆角按钮 */
                    font-size: 18px;
                    font-weight: bold;
                    padding: 12px 20px;  /* 增加内边距 */
                    min-width: 180px;  /* 最小宽度 */
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  /* 按钮阴影 */
                    transition: all 0.3s ease;  /* 动画过渡效果 */
                }

                QPushButton:hover {
                    background-color: #45a049;  /* 鼠标悬停时变色 */
                    transform: translateY(-2px);  /* 鼠标悬停时按钮上移 */
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);  /* 鼠标悬停时阴影加深 */
                }

                QPushButton:pressed {
                    background-color: #388E3C;  /* 按钮按下时颜色 */
                    transform: translateY(1px);  /* 按钮按下时微微下移 */
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);  /* 按钮按下时阴影减弱 */
                }
            """)

            # 为每个按钮绑定点击事件
            btn.clicked.connect(lambda _, text=btn_text: self.switch_page(text))
            self.sidebar.addWidget(btn)

    def create_pages(self):
        # 创建监测页面
        self.page_monitor = self.create_monitor_page()

        # 创建结果页面
        self.page_result = self.create_result_page()

        # 创建用户管理页面   先空置  回来添加数据库增加登陆功能
        self.page_user = QLabel("用户管理页面内容")

        # 创建图片下载页面
        self.page_download = self.create_image_page()

        # 创建数学函数图形化界面   先空置
        self.page_math = QLabel("数学函数图形化页面内容")

        # 添加页面到堆栈
        self.pages.addWidget(self.page_monitor)
        self.pages.addWidget(self.page_user)
        self.pages.addWidget(self.page_download)
        self.pages.addWidget(self.page_math)
        self.pages.addWidget(self.page_result)

    def create_monitor_page(self):
        # 创建监测功能页面
        page = QWidget()
        layout = QHBoxLayout(page)  # 正确实例化 QHBoxLayout

        # 左侧：上传图片和检测按钮
        self.image_label = QLabel("上传一张图片")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid #D1D1D1; border-radius: 8px; background-color: white;")

        self.upload_button = QPushButton("上传图片")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.upload_button.clicked.connect(self.upload_image)

        self.detect_button = QPushButton("开始检测")
        self.detect_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
                font-size: 16px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.detect_button.clicked.connect(self.run_detection)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        left_layout.addWidget(self.upload_button)
        left_layout.addWidget(self.detect_button)

        # 右侧：检测结果
        self.result_label = QLabel("检测结果将显示在此")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFixedSize(600, 480)
        self.result_label.setStyleSheet("border: 2px solid #D1D1D1; border-radius: 8px; background-color: white;")

        # 右侧：柱状图显示区域
        self.chart_label = QLabel("柱状图将显示在此")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setFixedSize(600, 480)
        self.chart_label.setStyleSheet("border: 2px solid #D1D1D1; border-radius: 8px; background-color: white;")

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.chart_label)  # 添加柱状图显示区域

        layout.addLayout(left_layout, 1)  # 使用正确的布局对象
        layout.addLayout(right_layout, 1)

        return page

# 上传图片
    def upload_image(self):
        # 打开文件对话框上传图片
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp)", options=options)

        if file:
            self.image_path = file
            pixmap = QPixmap(file)  # 加载图片

            if pixmap.isNull():
                print("无法加载图片")
            else:
                # 设置 QLabel 显示图片
                self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))  # 保持比例缩放
                self.image_label.setAlignment(Qt.AlignCenter)  # 图片居中显示
                self.image_label.setText("")  # 清空 QLabel 上的文件名文本

    def run_detection(self):
        if not hasattr(self, "image_path") or not self.image_path:
            self.image_label.setText("请先上传图片")
            return

        # 创建YOLOv5Detector对象
        detector = YOLOv5Detector(model_path=MODEL_PATH, device="cpu")
        output_path, detected_objects = detector.detect(self.image_path)

        results = detector.model(self.image_path)  # 重新运行推理，获取结果对象
        # 显示检测后的图片
        pixmap = QPixmap(output_path).scaled(640, 480, Qt.KeepAspectRatio)
        self.result_label.setPixmap(pixmap)

        category_counts = detector.extract_detected_objects(results)  # 提取类别计数
        # 可视化柱状图
        self.plot_bar_chart(detected_objects, category_counts)

        # 刷新画廊
        self.refresh_list()

    def plot_bar_chart(self, detected_objects, category_counts):
        """
        根据检测到的物体生成柱状图，横坐标为类别，纵坐标为每个类别的数量
        :param detected_objects: 检测到的物体列表，每个物体包含类别和置信度
        :param category_counts: 各类别的计数字典
        """
        if not detected_objects:
            return

        # 遍历每个检测结果，确保 detected_objects 是字典结构
        for obj in detected_objects:
            if isinstance(obj, dict) and "class" in obj:
                category_name = obj["class"].lower()  # 使类别名称小写，避免大小写差异
                category_counts[category_name] = category_counts.get(category_name, 0) + 1

        # 调试
        # print("Category Counts:", category_counts)

        # 提取类别和检测数量
        classes = list(category_counts.keys())
        counts = list(category_counts.values())

        # 创建柱状图
        plt.figure(figsize=(12, 8))  # 设置图表尺寸
        plt.bar(classes, counts, color='skyblue')

        # 添加标签  中文会乱码不知道为什么
        plt.xlabel("Item Category")
        plt.ylabel("Count")
        plt.title("Detected Items Count per Category")

        # 设置纵坐标显示为整数  MaxNLocator限制为整数
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

        # 获取检测图片的文件名，并生成新的图表文件名
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]  # 获取图片文件名（不带扩展名）
        result_image_dir = os.path.join(os.getcwd(), "result_image")  # 结果图像目录
        os.makedirs(result_image_dir, exist_ok=True)  # 确保目录存在

        # 保存图表到文件，命名为 "检测的文件名_label.png"
        chart_path = os.path.join(result_image_dir, f"{base_name}_label.png")
        plt.savefig(chart_path)

        # 关闭图表
        plt.close()

        # 显示柱状图
        pixmap = QPixmap(chart_path).scaled(640, 480, Qt.KeepAspectRatio)
        self.chart_label.setPixmap(pixmap)

    def create_result_page(self):
        # 创建主容器
        self.result_page = QWidget()
        layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("检测结果图片与柱状图")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title_label)

        # 创建滚动区域以支持图片与柱状图列表
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # 创建列表容器
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: white;
                border: 1px solid #D1D1D1;
                border-radius: 8px;
                padding: 10px;
            }
            QListWidget::item {
                padding: 20px;
                border-bottom: 1px solid #D1D1D1;
            }
        """)

        # 加载图片并刷新列表
        self.refresh_list()

        scroll_area.setWidget(self.list_widget)
        layout.addWidget(scroll_area)
        self.result_page.setLayout(layout)

        return self.result_page

    def refresh_list(self):
        # 清除旧的列表项
        self.list_widget.clear()

        # 获取 result_image 目录下以 "detected" 开头的所有图片
        result_dir = os.path.join(os.getcwd(), "result_image")
        os.makedirs(result_dir, exist_ok=True)  # 确保目录存在
        images = [os.path.join(result_dir, f) for f in os.listdir(result_dir) if
                  f.lower().startswith('detected') and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 动态生成每个图片和柱状图的列表项
        for idx, image_path in enumerate(images):
            base_name = os.path.basename(image_path)  # 获取图片文件名
            base_name_no_ext = os.path.splitext(base_name)[0]  # 获取文件名不带扩展名

            # 构建对应的柱状图路径
            chart_path = os.path.join(os.getcwd(), "result_image", f"{base_name_no_ext.split('_')[1]}_label.png")

            # 如果柱状图存在，创建列表项
            if os.path.exists(chart_path):
                item = QListWidgetItem()
                item.setSizeHint(QSize(1000, 500))  # 增大每个列表项的高度

                # 创建一个水平布局来放置序号、图片、柱状图、删除按钮以及课堂信息
                widget = QWidget()
                layout = QHBoxLayout()

                # 序号标签
                index_label = QLabel(f"{idx + 1}. {base_name.split('_')[1].split('.')[0]}")
                index_label.setAlignment(Qt.AlignCenter)
                index_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-right: 20px;")
                layout.addWidget(index_label)

                # 检测图片
                image_label = QLabel()
                pixmap = QPixmap(image_path).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(pixmap)
                image_label.setStyleSheet("border: 2px solid #D1D1D1; border-radius: 8px; padding: 5px;")

                # 柱状图
                chart_label = QLabel()
                chart_pixmap = QPixmap(chart_path).scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                chart_label.setPixmap(chart_pixmap)
                chart_label.setStyleSheet("border: 2px solid #D1D1D1; border-radius: 8px; padding: 5px;")

                # 课堂总人数和课堂评价   这个人数未来通过添加数据库来进行存储，会比较方便。没怎么用过python的数据库操作，之前都是用java
                total_students_label = QLabel("10(测试)")
                total_students_label.setAlignment(Qt.AlignCenter)
                total_students_label.setStyleSheet("font-size: 14px; color: #333; margin-left: 20px;")
                layout.addWidget(total_students_label)

                # 这个评价跟人数一样，也是考虑通过数据库来存储。回来试试接入chatgpt来生成评价，
                # 不过之前大一的时候申请chatgpt的token需要用到美国手机号，当时试了很久也没有成功，回来再看看，先搁置一下
                class_feedback_label = QLabel("评价: 好(测试)")
                class_feedback_label.setAlignment(Qt.AlignCenter)
                class_feedback_label.setStyleSheet("font-size: 14px; color: #333; margin-left: 20px;")
                layout.addWidget(class_feedback_label)

                # 删除按钮  样式就这样了……
                delete_button = QPushButton("删除")
                delete_button.setStyleSheet("""
                    QPushButton {
                        background-color: #FF5252;
                        color: white;
                        font-size: 18px;
                        padding: 10px 15px;
                        border-radius: 8px;
                        min-width: 50px;
                    }
                    QPushButton:hover {
                        background-color: #E53935;
                    }
                """)
                delete_button.clicked.connect(   # 绑定按钮事件
                    lambda checked, image=image_path, chart=chart_path: self.confirm_delete(image, chart))

                # 将序号，文件名，评论图片，柱状图和删除按钮添加到布局中
                layout.addWidget(image_label)
                layout.addWidget(chart_label)
                layout.addWidget(delete_button)

                # 将布局应用到一个新创建的widget上
                widget.setLayout(layout)

                # 设置每个列表项的内容
                item.setData(Qt.UserRole, (image_path, chart_path))  # 保存路径信息
                self.list_widget.addItem(item)
                self.list_widget.setItemWidget(item, widget)

    def confirm_delete(self, image_path, chart_path):
        """
        弹出确认删除的提示框，用户确认后删除图片和柱状图
        """
        reply = QMessageBox.question(self, "确认删除", "确定删除此项记录吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            self.delete_record(image_path, chart_path)

    def delete_record(self, image_path, chart_path):
        """
        删除图片和柱状图文件以及对应的列表项
        """
        # 删除文件
        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(chart_path):
            os.remove(chart_path)

        # 刷新列表，删除对应的记录
        self.refresh_list()



    def create_user_page(self):

        pass

    def create_image_page(self):
        # 创建图片下载页面
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(20)  # 增加控件之间的间距，避免过于紧凑

        # 添加标题
        title_label = QLabel("Pixabay 图片下载工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 30px; 
            font-weight: bold; 
            color: #333; 
            margin-bottom: 20px;
        """)
        layout.addWidget(title_label)

        # 关键词输入框
        keyword_label = QLabel("关键词：")
        keyword_label.setStyleSheet("""
            font-size: 18px; 
            color: #666; 
            font-weight: 600;
        """)
        self.keyword_input = QLineEdit(self)
        self.keyword_input.setPlaceholderText("请输入关键词（例如：cat）")
        self.keyword_input.setStyleSheet("""
            font-size: 16px; 
            padding: 12px;
            background-color: #f1f1f1;
            border-radius: 10px;
            border: 1px solid #ddd;
        """)
        layout.addWidget(keyword_label)
        layout.addWidget(self.keyword_input)

        # 图片数量输入框
        num_label = QLabel("图片数量：")
        num_label.setStyleSheet("""
            font-size: 18px; 
            color: #666; 
            font-weight: 600;
        """)
        self.num_input = QLineEdit(self)
        self.num_input.setPlaceholderText("请输入图片数量")
        self.num_input.setStyleSheet("""
            font-size: 16px; 
            padding: 12px;
            background-color: #f1f1f1;
            border-radius: 10px;
            border: 1px solid #ddd;
        """)
        layout.addWidget(num_label)
        layout.addWidget(self.num_input)

        # 下载按钮
        download_button = QPushButton("下载图片")
        download_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 12px;
                padding: 12px 20px;
                font-size: 18px;
                font-weight: bold;
                border: none;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                transition: background-color 0.3s, transform 0.2s;
            }
            QPushButton:hover {
                background-color: #45A049;
                transform: translateY(-3px);
            }
            QPushButton:pressed {
                background-color: #388E3C;
                transform: translateY(1px);
            }
        """)
        download_button.clicked.connect(self.start_download)
        layout.addWidget(download_button, alignment=Qt.AlignCenter)

        # 提示信息
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            font-size: 16px;
            color: #333;
            font-weight: 400;
        """)
        layout.addWidget(self.status_label)

        # 进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #f1f1f1;
                border-radius: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 确保布局充满整个页面
        page.setLayout(layout)
        return page

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

        # 启动下载线程
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

# 启动
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())