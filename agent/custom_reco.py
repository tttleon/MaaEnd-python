import json
import random
import threading
import time
from warnings import catch_warnings

from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.pipeline import JActionType, JSwipe, JLongPress, JRecognitionType, JNeuralNetworkDetect, JTemplateMatch, \
    JClick, JOCR, JCustomRecognition

from agent.config import get_resource_path
from agent.utils.util import *
from agent.utils.yolo_det import *


class YoloDet(CustomRecognition):
    # 定义一个中文别名
    RECO_NAME = "YOLODet_cust"

    def analyze(
            self,
            context: Context,
            argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        controller = context.tasker.controller
        custom_recognition_param = json.loads(argv.custom_recognition_param)
        model = custom_recognition_param["model"]
        img = yolo_img_pre(argv.image)
        model_path = f'{get_resource_path()}/model/detect/{model}'
        detector = YOLODet(model_path, conf_thres=0.4)

        # 检测目标（返回边界框、分数、类别ID、掩码）
        boxes, scores, class_ids = detector(img)
        # [x1, y1, x2, y2] -> [x1, y1, w, h]
        boxes = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes]
        # 批量转换
        boxes = [restore_box(b) for b in boxes]
        # 640*640 rescale to 1280*720
        scores = scores.tolist()
        class_ids = class_ids.tolist()

        # cv2 draw
        # for box, score, class_id in zip(boxes, scores, class_ids):
        #     # 解构你的 [x1, y1, w, h]
        #     x1, y1, w, h = [int(v) for v in box]  # OpenCV 函数需要整数坐标
        #
        #     # 计算右下角坐标
        #     x2, y2 = x1 + w, y1 + h
        #
        #     # 设定颜色 (BGR)，这里根据类别 ID 随机选色或固定颜色
        #     color = (0, 255, 0)  # 绿色
        #
        #     # 1. 画矩形框
        #     cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness=2)
        #
        #     # 2. 准备标签文本
        #     label = f"ID:{class_id} {score:.2f}"
        #
        #     # 3. 在框上方画一个背景填充矩形，让文字更清晰
        #     (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        #     cv2.rectangle(img_draw, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        #
        #     # 4. 写上文字
        #     cv2.putText(img_draw, label, (x1, y1 - 5),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # cv2.imwrite("img_draw.png",img_draw)

        # 打印结果
        # print("Boxes:", boxes)
        # print("Scores:", scores)
        # print("Class IDs:", class_ids)

        # time.sleep(100)

        best_score_box = None
        best_result = None
        # 获取最高分索引
        if len(boxes) != 0:
            # 找到 scores 列表中的最大值，并获取它在列表中的索引
            best_score_index = scores.index(max(scores))
            best_score_box = boxes[best_score_index]
            best_result = {
                "box": best_score_box,
                "score": scores[best_score_index],
                "class_id": class_ids[best_score_index],
            }

        return CustomRecognition.AnalyzeResult(
            box=best_score_box,
            detail={"best_result": best_result, "boxes": boxes, "scores": scores, "class_ids": class_ids}
        )







