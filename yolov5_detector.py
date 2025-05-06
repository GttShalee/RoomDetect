import torch
import os
from PIL import Image

class YOLOv5Detector:
    def __init__(self, model_path, device="cpu"):
        """
        有个注意的是，调用load函数时请加上source="local"不然她会
        提示你什么从github上下载带宽不够云云，其实只要下载到本地就行了，不用在线下载
        什么时候能用gpu跑一下🤓
        """
        self.device = device
        # 加载 YOLOv5 模型

        self.model = torch.hub.load("./yolov5", "custom", path=model_path, source="local")
        self.model.eval()

    def detect(self, img_path):
        # 运行检测
        results = self.model(img_path)
        results.render()  # 给图片加上标注

        # 获取检测到的物体信息
        detected_objects = self.extract_detected_objects(results)

        # 保存结果图片
        result_dir = os.path.join(os.getcwd(), "result_image")
        os.makedirs(result_dir, exist_ok=True)  # 注意路径得有足够权限
        image_name = os.path.basename(img_path)
        output_path = os.path.join(result_dir, f"detected_{image_name}")  # 格式化文件名，方便后面操作
        result_image = results.ims[0]  # 使用 ims 代替 imgs
        Image.fromarray(result_image).save(output_path)

        return output_path, detected_objects  # 返回保存的图片路径和检测到的物体信息

    # 解析检测物品的函数
    def extract_detected_objects(self, results):
        """
        提取每个类别的检测数量
        :参数注释
        :results: YOLOv5的检测结果对象  具体到这个代码 => results = detector.model(self.image_path)
        :return: category_counts 每个类别的检测数量字典
        """
        category_counts = {}  # 先初始化

        # 遍历检测结果中的类别标签
        for detection in results.xywh[0]:  # 这里遍历每个检测框
            # 结果的格式是 [x_center, y_center, width, height, confidence, class]
            class_id = int(detection[-1])  # 获取类别标签（class_id）
            category_name = results.names[class_id]  # 获取类别名称

            # 更新类别计数
            if category_name in category_counts:
                category_counts[category_name] += 1
            else:
                category_counts[category_name] = 1

        # print("Category counts:", category_counts)

        category_sum = 0
        for category_name in category_counts:
            category_sum += category_counts[category_name]

        return category_counts