import cv2
import numpy as np
import onnxruntime
import math
import time
import random

# 设置随机种子，让颜色生成固定（可选）
random.seed(42)
np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def yolo_img_pre(image, input_size=(640, 640), bg_color=(114, 114, 114)):
    """
    预处理图像用于模型输入
    将图像按比例缩放后居中放置在指定颜色的背景图上，避免拉伸变形

    Args:
        image: 输入的原始图像 (numpy array)
        input_size: 目标尺寸 (width, height)，默认(640, 640)
        bg_color: 背景颜色，默认(114, 114, 114)

    Returns:
        img: 处理后的图像 (640x640)
    """
    # 获取原始图像的尺寸
    h, w = image.shape[:2]
    target_w, target_h = input_size

    # 计算缩放比例（取宽和高中较小的比例，保证图像完整显示）
    scale = min(target_w / w, target_h / h)

    # 按比例缩放图像
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建指定颜色的背景图
    img_padded = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)

    # 计算居中放置的坐标
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # 将缩放后的图像粘贴到背景图的中心位置
    img_padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    return img_padded




class YOLODet:
    def __init__(self, path, conf_thres=0.4, num_masks=32):
        """
        Args:
            path (str): Path to the exported ONNX model.
            conf_thres (float): Confidence threshold for filtering detections.
            num_masks (int): Number of mask coefficients (should match export, e.g., 32).
        """
        self.conf_threshold = conf_thres
        self.num_masks = num_masks
        self.initialize_model(path)
        # COCO数据集80个类别的名称（可根据你的模型修改）
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        # 为每个类别生成随机颜色
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]

    def initialize_model(self, path):
        # Create ONNX Runtime session with GPU (if available) or CPU.
        # self.session = onnxruntime.InferenceSession(
        #     path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        # )
        self.session = onnxruntime.InferenceSession(
            path, providers=['CPUExecutionProvider']
        )
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in model_inputs]
        self.input_shape = model_inputs[0].shape  # Expected shape: (1, 3, 640, 640)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [out.name for out in model_outputs]

    def prepare_input(self, im):
        # 获取图片尺寸
        self.img_height, self.img_width = im.shape[:2]

        # 1. BGR转RGB（cv2默认BGR，模型通常期望RGB）
        im = im[..., ::-1]

        # 2. HWC (高度, 宽度, 通道) 转 CHW (通道, 高度, 宽度)
        im = im.transpose((2, 0, 1))

        # 3. 确保数组内存连续（提升后续计算效率）
        im = np.ascontiguousarray(im)

        # 4. 转换为float32类型并归一化到0-1范围
        im = im.astype(np.float32) / 255.0

        # 5. 添加batch维度（从(3, H, W)转为(1, 3, H, W)，适配模型输入）
        im = np.expand_dims(im, axis=0)

        # # 2. 遍历每个通道（假设batch_size=1）
        # for channel_idx in range(3):
        #     # 提取对应通道：取第0个batch，第channel_idx个通道的所有高宽数据
        #     channel_data = im[0, channel_idx, :, :]
        #     # 3. 将张量转为numpy数组，方便保存为文本
        #     channel_np = channel_data
        #     # 4. 定义文件名
        #     filename = f"im_channel_{channel_idx}.txt"
        #     # 5. 保存为文本文件（每行一个行数据，元素用空格分隔）
        #     np.savetxt(filename, channel_np, fmt='%.6f')  # fmt控制输出精度
        #     print(f"已保存通道 {channel_idx} 数据到 {filename}")

        return im

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def segment_objects(self, image):
        """
        Processes an image and returns:
          - boxes: Bounding boxes (rescaled to original image coordinates).
          - scores: Confidence scores.
          - class_ids: Detected class indices.
          - masks: Binary segmentation masks (aligned with the original image).
        """
        # Preprocess the image.
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)


        # 打印 outputs长度
        # print(len(outputs)) # 1
        # print(outputs[0].shape) #  # (1, 300, 6)

        # Process detection output.
        detections = np.squeeze(outputs[0], axis=0)  # Now shape: (300, 38)

        # Filter out detections below the confidence threshold.
        valid = detections[:, 4] > self.conf_threshold
        detections = detections[valid]
        #
        # if detections.shape[0] == 0:
        #     return np.array([]), np.array([]), np.array([]), np.array([])

        # Extract detection results.
        # boxes_model: boxes in model input coordinates (e.g., in a 640x640 space)
        boxes_model = detections[:, :4]  # Format: (x1, y1, x2, y2)
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(np.int64)

        # 打印第二个检测框的信息
        # print("第二个检测框信息:")
        # print(f"框坐标: {boxes_model[1]}")
        # print(f"置信度: {scores[1]}")
        # print(f"类别ID: {class_ids[1]}")
        # print(f"掩码系数: {mask_coeffs[1]}")

        # Rescale boxes for final drawing on the original image.
        boxes_draw = self.rescale_boxes(
            boxes_model,
            (self.input_height, self.input_width),
            (self.img_height, self.img_width)
        )

        # return boxes_draw.tolist(), scores.tolist(), class_ids.tolist()
        return boxes_draw, scores, class_ids



    @staticmethod
    def rescale_boxes(boxes, input_shape, target_shape):
        """
        Rescales boxes from one coordinate space to another.

        Args:
            boxes (np.ndarray): Array of boxes (N, 4) with format [x1, y1, x2, y2].
            input_shape (tuple): (height, width) of the current coordinate space.
            target_shape (tuple): (height, width) of the target coordinate space.

        Returns:
            np.ndarray: Scaled boxes of shape (N, 4).
        """
        in_h, in_w = input_shape
        tgt_h, tgt_w = target_shape
        scale = np.array([tgt_w / in_w, tgt_h / in_h, tgt_w / in_w, tgt_h / in_h])
        return boxes * scale

    def draw_results(self, image, boxes, scores, class_ids, masks):
        """
        在原始图像上绘制检测结果（边界框、标签、掩码）
        Args:
            image (np.ndarray): 原始图像
            boxes (np.ndarray): 边界框坐标 (N,4) [x1,y1,x2,y2]
            scores (np.ndarray): 置信度分数 (N,)
            class_ids (np.ndarray): 类别ID (N,)
            masks (np.ndarray): 分割掩码 (N, H, W)

        Returns:
            np.ndarray: 绘制后的图像
        """
        # 创建图像副本，避免修改原图
        draw_img = image.copy()

        if len(boxes) == 0:
            return draw_img

        # # 绘制分割掩码（半透明效果）
        # alpha = 0.5  # 掩码透明度
        # for i in range(len(boxes)):
        #     class_id = class_ids[i]
        #     color = self.colors[class_id]
        #
        #     # 获取掩码并转换为彩色
        #     mask = masks[i]
        #     # 1. 创建和原图同尺寸的彩色掩码图
        #     color_mask = np.zeros_like(draw_img)
        #     color_mask[mask == 1] = color
        #
        #     # 2. 关键修改：只在掩码区域执行加权融合，非掩码区域保持原图不变
        #     # 先创建掩码的布尔索引
        #     mask_index = mask == 1
        #     # 仅对掩码覆盖的像素进行叠加计算
        #     draw_img[mask_index] = cv2.addWeighted(
        #         draw_img[mask_index].reshape(-1, 3), 1 - alpha,
        #         color_mask[mask_index].reshape(-1, 3), alpha, 0
        #     )

        # 绘制边界框和标签
        thickness = 2  # 边界框线宽
        font_scale = 0.6  # 字体大小
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = scores[i]
            class_id = class_ids[i]
            class_name = self.class_names[class_id]
            color = self.colors[class_id]

            # 绘制边界框
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, thickness)

            # 绘制标签背景
            label = f"{class_name}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_height + 10

            # 绘制标签背景矩形
            cv2.rectangle(draw_img, (x1, label_y - label_height - 5),
                          (x1 + label_width, label_y + 5), color, -1)

            # 绘制标签文字
            cv2.putText(draw_img, label, (x1, label_y), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        return draw_img

    def __call__(self, image):
        # This allows you to call the instance directly, e.g.:
        # boxes, scores, class_ids, masks = detector(image)
        return self.segment_objects(image)


if __name__ == "__main__":
    # 配置
    # Load the model and create InferenceSession
    best_weights_path = r"E:\myJobTwo\project\ultralytics\m_train\endfield\runs\detect\endfield-fight2\weights\fight-op15-nms.onnx"

    detector = YOLODet(best_weights_path, conf_thres=0.4)

    # 读取图像
    img = cv2.imread(r"E:\myJobTwo\project\ultralytics\m_train\endfield\test.png")
    if img is None:
        print("错误：无法读取图像文件，请检查路径是否正确！")
        exit()
    # 预处理图像
    img = yolo_img_pre(img)

    # 检测目标（返回边界框、分数、类别ID、掩码）
    boxes, scores, class_ids = detector(img)
    draw_img = detector.draw_results(img, boxes, scores, class_ids, None)
    # cv2.imwrite("draw_img.png", draw_img)


    # 打印结果
    print("Boxes:", boxes)
    print("Scores:", scores)
    print("Class IDs:", class_ids)
