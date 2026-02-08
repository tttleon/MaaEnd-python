import json
import random
import threading
import time

import numpy as np
from maa.custom_action import CustomAction
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JCustomRecognition, JRecognitionType, JLongPress, JActionType, JSwipe, JClick, JOCR, \
    JTemplateMatch

from agent.utils.util import *


class OpenMap(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "打开地图_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("打开地图")
        return check_task_success(res)

class ResizeMapToMin(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "地图比例缩到最小_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("地图比例缩到最小")
        return check_task_success(res)


class MapBigMoveToLeft(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "地图左移到底_comp"
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("地图左移到底")
        return check_task_success(res)

class MapBigMoveToRight(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "地图右移到底_comp"
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("地图右移到底")
        return check_task_success(res)

# 上移到底
class MapBigMoveToTop(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "地图上移到底_comp"
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("地图上移到底")
        return check_task_success(res)

# 下移到底
class MapBigMoveToBottom(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "地图下移到底_comp"
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("地图下移到底")
        return check_task_success(res)


class OpenAreaOverview(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "点击地区总览_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("点击地区总览")
        return check_task_success(res)

class OriginiumResearchInstituteMapClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "源石研究院大地图点击_comp"

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("源石研究院大地图点击")
        return check_task_success(res)

#枢纽区大地图点击
class HubAreaMapClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "枢纽区大地图点击_comp"

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("枢纽区大地图点击")
        return check_task_success(res)

#协议传送点点击
class ProtocolTransferPointClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "协议传送点点击_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("协议传送点点击")
        return check_task_success(res)

#源石研究院-研究所上传送点
class OriginiumResearchInstituteTransferPointClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "源石研究院-研究所上传送点_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("源石研究院-研究所上传送点")
        return check_task_success(res)
#猫头鹰点击收取资源
class OriginiumResearchInstituteOwlClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "猫头鹰点击收取资源_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("猫头鹰点击收取资源")
        return check_task_success(res)

#加载中界面判断
class LoadingScreenCheck(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "加载中界面判断_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("加载中界面判断")
        return check_task_success(res)
# 枢纽区-基地电站传送点
class HubAreaBasePowerStationTransferPointClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "枢纽区-基地电站传送点_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("枢纽区-基地电站传送点")
        return check_task_success(res)

# 枢纽区-工人之家传送点
class HubAreaWorkerHouseTransferPointClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "枢纽区-工人之家传送点_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("枢纽区-工人之家传送点")
        return check_task_success(res)

#谷地通道大地图点击
class ValleyChannelMapClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "谷地通道大地图点击_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("谷地通道大地图点击")
        return check_task_success(res)

#谷地通道-通道入口传送点
class ValleyChannelTransferPointClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "谷地通道-通道入口传送点_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("谷地通道-通道入口传送点")
        return check_task_success(res)

#阿伯莉采石场大地图点击
class AblStoneBigMapClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "阿伯莉采石场大地图点击_comp"

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("阿伯莉采石场大地图点击")
        return check_task_success(res)

#矿脉源区大地图点击
class MineralMineSourceBigMapClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "矿脉源区大地图点击_comp"
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("矿脉源区大地图点击")
        return check_task_success(res)

#矿脉源区-医疗站传送点
class MineralMineSourceMedicalStationTransferPointClick(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "矿脉源区-医疗站传送点_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res= context.run_task("矿脉源区-医疗站传送点")
        return check_task_success(res)

class blackScreenCheckFunc(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "黑屏判断Func_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        controller = context.tasker.controller
        # cv2 平均灰度值
        screen_cap = controller.post_screencap().wait().get()
        gray = cv2.cvtColor(screen_cap, cv2.COLOR_BGR2GRAY)
        avg_gray = np.mean(gray)
        if avg_gray < 10:
            return True
        return False

#黑屏判断
class blackScreenCheck(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "黑屏判断_comp"
    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> bool:
        res = context.run_task("黑屏判断")
        return check_task_success(res)


class FightEndReco(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "协议空间战斗结束触摸烟雾_Func"
    print("FightEndReco")
    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        controller = context.tasker.controller
        screen_center_x = 1280 // 2
        start_x = screen_center_x + 100
        screen_center_y = 720 // 2

        w_x = 192
        w_y = 441
        # 1. 定义方向坐标映射
        directions = {
            'W': (192, 441),
            'A': (78, 552),
            'S': (192, 647),
            'D': (290, 555)
        }

        yolo_none_count = 0
        while True:
            recognition_config = JCustomRecognition(
                custom_recognition="YOLODet_cust",
                custom_recognition_param={"model": "fight-nms.onnx"}
            )

            screen_cap = controller.post_screencap().wait().get()
            yolo_res = context.run_recognition_direct(
                JRecognitionType.Custom,
                recognition_config,
                screen_cap
            )

            if yolo_res.best_result is None:
                print(f'yolo_best_result none!')
                yolo_none_count += 1
                if yolo_none_count >= 6:
                    print('yolo_none_count >= 6, 视角转了一圈都没找到，尝试随机移动')
                    # 2. 随机选择一个方向
                    dir_name, (target_x, target_y) = random.choice(list(directions.items()))

                    # 3. 随机生成持续时间（比如 1000ms 到 2000ms 之间）
                    random_duration = random.randint(1000, 2000)

                    print(f'向 [{dir_name}] 方向行走 {random_duration}ms')

                    action_param = JLongPress(
                        target=(target_x, target_y, 0, 0),
                        duration=random_duration,
                    )

                    context.run_action_direct(JActionType.LongPress, action_param)
                    time.sleep(0.3)
                    yolo_none_count = 0
                    continue
                action_param = JSwipe(
                    begin=(start_x, screen_center_y, 0, 0),
                    duration=[500],
                    end=[(start_x + 200, screen_center_y, 0, 0)],
                )
                context.run_action_direct(JActionType.Swipe, action_param)
                continue

            my_detail = yolo_res.best_result.detail
            print(f'yolo detail={my_detail}')
            # 拿到具体的值
            yolo_best_result = my_detail.get("best_result")
            # print(yolo_best_result)
            # scores = my_detail.get("scores")
            # boxes = my_detail.get("boxes")
            score = yolo_best_result["score"]
            box = yolo_best_result["box"]
            print(f'score={box}')
            print(f'score={score}')
            if score < 0.5:
                action_param = JSwipe(
                    begin=(start_x, screen_center_y, 0, 0),
                    duration=[500],
                    end=[(start_x + 200, screen_center_y, 0, 0)],
                )
                context.run_action_direct(JActionType.Swipe, action_param)
                continue

            yolo_center_x = box[0] + box[2] // 2
            # 863 135 64 22
            yolo_top = box[1]
            print(f'yolo_center_x={yolo_center_x}')

            # 如果检测中心点在屏幕中间左侧100像素之外
            if yolo_center_x < screen_center_x - 150:
                action_param = JSwipe(
                    begin=(start_x, screen_center_y, 0, 0),
                    duration=[500],
                    end=[(start_x - 50, screen_center_y, 0, 0)],
                )
                context.run_action_direct(JActionType.Swipe, action_param)
                print('在屏幕左侧')
                # time.sleep(100)
            # 如果检测中心点在屏幕中间右侧100像素之外
            elif yolo_center_x > screen_center_x + 150:
                action_param = JSwipe(
                    begin=(start_x, screen_center_y, 0, 0),
                    duration=[500],
                    end=[(start_x + 50, screen_center_y, 0, 0)],
                )
                print('在屏幕右侧')
                context.run_action_direct(JActionType.Swipe, action_param)
            else:
                print('在屏幕靠中间')

                action_param = JLongPress(
                    target=(w_x, w_y, 0, 0),
                    duration=700,
                )
                print('走起')
                context.run_action_direct(JActionType.LongPress, action_param)
                time.sleep(0.3)

                ocr_param = JOCR(
                    expected=["领取奖励"],
                    roi=(765, 403, 103, 41),
                )
                screen_cap = controller.post_screencap().wait().get()
                res = context.run_recognition_direct(JRecognitionType.OCR, ocr_param, screen_cap)
                print(res)
                if res.best_result:
                    score = res.best_result.score
                    text = res.best_result.text
                    box = res.best_result.box
                    if score > 0.9 and text == "领取奖励":
                        # click
                        click_x = box[0] + box[2] // 2
                        click_y = box[1] + box[3] // 2
                        action_param = JClick(
                            target=(click_x, click_y, 0, 0),
                        )
                        context.run_action_direct(JActionType.Click, action_param)
                        time.sleep(0.5)
                        break
        return True

class FightReco(CustomAction):
    ACTION_NAME = "协议空间战斗_Func"

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        try:
            controller = context.tasker.controller

            state = {"stop_flag": False, "no_text_count": 0, "is_focus_clicked": False}
            def lock_check():
                while not state["stop_flag"]:
                    screen_cap = controller.post_screencap().wait().get()
                    lock_icon_roi = (808, 637, 22, 22)
                    lock_icon_img = screen_cap[lock_icon_roi[1]:lock_icon_roi[1]+lock_icon_roi[3], lock_icon_roi[0]:lock_icon_roi[0]+lock_icon_roi[2]]
                    # 平均灰度值
                    lock_icon_gray = np.mean(lock_icon_img)
                    print(f'锁定图标的平均灰度值是={lock_icon_gray}')
                    if lock_icon_gray < 150:
                        click_param = JClick(target=(815, 639, 10, 10))
                        context.run_action_direct(JActionType.Click, click_param)
                    # 10s 检查一次
                    time.sleep(10)

            # --- 1. OCR 检测线程 ---
            def ocr_worker():
                # 等个先打一会
                time.sleep(30)
                ocr_param = JOCR(expected=[], roi=(269, 15, 71, 40))
                while not state["stop_flag"]:
                    screen_cap = controller.post_screencap().wait().get()
                    res = context.run_recognition_direct(JRecognitionType.OCR, ocr_param, screen_cap)
                    if res.best_result and res.best_result.score >= 0.7:
                        state["no_text_count"] = 0
                    else:
                        state["no_text_count"] += 1

                    if state["no_text_count"] >= 3:
                        state["stop_flag"] = True
                    time.sleep(1)

            # --- 2. 技能1-4 随机点击线程 ---
            def skills_14_worker():
                skill_params = [
                    JClick(target=(919, 609, 30, 30)),  # 技能1
                    JClick(target=(900, 508, 30, 30)),  # 技能2
                    JClick(target=(961, 421, 30, 30)),  # 技能3
                    JClick(target=(1049, 397, 30, 30))  # 技能4
                ]
                while not state["stop_flag"]:
                    # 随机挑选一个技能执行
                    target_skill = random.choice(skill_params)
                    context.run_action_direct(JActionType.Click, target_skill)
                    # 执行后固定休眠 2s
                    time.sleep(2)

            # --- 3. LongPress 长按线程 ---
            def long_press_worker():
                fight_btn_param = JLongPress(target=(1029, 527, 54, 54), duration=4000)
                while not state["stop_flag"]:
                    context.run_action_direct(JActionType.LongPress, fight_btn_param)
                    # 执行后固定休眠 1s
                    time.sleep(1)

            # --- 4. ChainSkill 连携技线程 ---
            def chain_skill_worker():
                chain_skill_param = JClick(target=(857, 404, 30, 30))
                while not state["stop_flag"]:
                    context.run_action_direct(JActionType.Click, chain_skill_param)
                    # 执行后固定休眠 1s
                    time.sleep(0.5)

            # --- 启动所有线程 ---
            threads = [
                threading.Thread(target=lock_check, daemon=True),
                threading.Thread(target=ocr_worker, daemon=True),
                threading.Thread(target=skills_14_worker, daemon=True),
                threading.Thread(target=long_press_worker, daemon=True),
                threading.Thread(target=chain_skill_worker, daemon=True),
            ]

            for t in threads:
                t.start()

            # 主线程阻塞，等待结束标志
            while not state["stop_flag"]:
                time.sleep(0.5)


            return True
        except Exception as e:
            print(f"发生了其他未知的错误: {e}")
            return False

class WalkByPointsReco(CustomAction):
    # 定义一个中文别名
    ACTION_NAME = "根据提供点进行人物移动_Func"

    map_center_x = int(1280 / 2)
    map_center_y = int(720 / 2)
    rotate_start_x = map_center_x + 100
    # 前进摇杆点位
    w_x = 192
    w_y = 441

    def __init__(self):
        super().__init__()
        self.context = None

    def run(
            self,
            context: Context,
            argv: CustomAction.RunArg,
    ) -> bool:
        controller = context.tasker.controller
        custom_action_param = json.loads(argv.custom_action_param)

        self.context = context

        print('WalkByPointsReco')
        for task in custom_action_param:
            # 模版图片名称
            template_img_name = task["template_img_name"]
            # 地图滑动方向
            direction = task["direction"]
            # 是否需要调整地图大小
            black_arrow_need_resize = task.get("black_arrow_need_resize", False)
            # 是否是在帝江号上面，给控制中枢地图导航使用，不然每次都要判断在哪个地图上太耗时间了
            is_DJH = task.get("is_DJH", False)
            # 目标点列表
            points = task["points"]
            # 移动时地图要移动多少偏移量
            map_x_move_offset = task["map_x_move_offset"]  # 范围+-500
            map_y_move_offset = task["map_y_move_offset"]  # 范围+-500

            last_time_point = None
            for target_point in points:
                # 直接行走，不做任何计算
                if 'direct_walk' in target_point:
                    action_param = JLongPress(
                        target=(self.w_x, self.w_y, 0, 0),
                        duration=target_point["direct_walk"],
                    )
                    context.run_action_direct(JActionType.LongPress, action_param)
                    continue

                # 如果单个目标点需要不同的地图位移来调整箭头位置的的话，覆盖掉全局的
                if "map_x_move_offset" in target_point:
                    map_x_move_offset = target_point["map_x_move_offset"]  # 范围+-600
                if "map_y_move_offset" in target_point:
                    map_y_move_offset = target_point["map_y_move_offset"]  # 范围+-600

                # 打开地图
                self._open_map(is_DJH,map_x_move_offset,map_y_move_offset)


                # 先使用 YOLO 检测箭头在图片中的位置
                screen_cap = controller.post_screencap().wait().get()

                recognition_config = JCustomRecognition(
                    custom_recognition="YOLODet_cust",
                    custom_recognition_param={"model": "arrow-nms.onnx"}
                )

                yolo_res = context.run_recognition_direct(
                    JRecognitionType.Custom,
                    recognition_config,
                    screen_cap
                )
                print('yolo_res', yolo_res)
                my_detail = yolo_res.best_result.detail
                yolo_best_result = my_detail.get("best_result")
                box = [int(round(v)) for v in yolo_best_result["box"]]
                yolo_center_x = box[0] + box[2] // 2
                yolo_center_y = box[1] + box[3] // 2

                # 模版匹配，获取模板的左上角坐标作为坐标原点
                recognition_config = JTemplateMatch(
                    template=[template_img_name],
                )
                template_res = context.run_recognition_direct(
                    JRecognitionType.TemplateMatch,
                    recognition_config,
                    screen_cap
                )

                template_best_result = max(template_res.all_results, key=lambda x: x.score)
                top_left_x = template_best_result.box[0]
                top_left_y = template_best_result.box[1]

                # 计算箭头中心点相对于模板左上角 (top_left) 的坐标
                relative_x = yolo_center_x - top_left_x
                relative_y = yolo_center_y - top_left_y
                # 存储上一个点
                if last_time_point is None:
                    last_time_point = target_point
                else:
                    # 有上一个点，计算和上一个点之间的距离
                    to_last_distance = math.sqrt((relative_x - last_time_point["x"]) ** 2 + (relative_y - last_time_point["y"]) ** 2)
                    print(f"和上一个点之间的距离: {to_last_distance}")
                    last_time_point = target_point
                    # 没有到达点位附近，失败，重头开始跑吧
                    if to_last_distance > 20:
                        # 退出地图
                        context.run_action("点击右上角叉叉")
                        time.sleep(3)
                        return False


                # 移动地图使得箭头被黑圈圈包围，方便后续方向计算
                if black_arrow_need_resize:
                    # 帝江号纯移动地图没法变成黑圈，要先放大地图到最大
                    context.run_action("放大地图到最大")

                if direction == 'top':
                    y_start = 720 - 10
                    y_end = y_start - 500
                else:
                    # 你妈的怎么还把手机状态栏给滑下来了
                    y_start = 0 + 60
                    y_end = y_start + 500
                # 移动地图使得箭头在地图边缘被黑圈包围
                action_param = JSwipe(
                    begin=(10, y_start, 0, 0),
                    duration=[700],
                    end=[(10, y_end, 0, 0)],
                )
                print("移动地图使得箭头在地图边缘被黑圈包围")
                context.run_action_direct(JActionType.Swipe, action_param)
                context.run_action_direct(JActionType.Swipe, action_param)

                # 再次截图
                screen_cap = controller.post_screencap().wait().get()
                # 退出地图
                context.run_action("点击右上角叉叉")

                # 切出箭头图标 上滑，下滑y值是固定的，x值通过yolo检测结果box的中心来确定
                if direction == 'top':
                    y = 121
                elif direction == 'bottom':
                    y = 563
                else:
                    raise ValueError("direction must be 'top' or 'bottom'")
                x = box[0]
                w = box[2]
                h = box[3] + 10
                cut_arrow = screen_cap[y:y + h, x:x + w]
                # cv2.imwrite("cut_arrow.png",cut_arrow)
                # 灰度二值图
                gray = cv2.cvtColor(cut_arrow, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
                # 计算自身角度
                arrow_angle = get_angle_and_bisector(thresh)
                # 计算自身角度失败，重试
                if arrow_angle is False:
                    return False


                # 计算到目标点需要旋转的角度
                target_x = target_point['x']
                target_y = target_point['y']
                rotate_angle = get_angle_to_target(relative_x, relative_y, arrow_angle, target_x, target_y)
                # 计算需要旋转的像素距离
                angle_pix_rate = 30 / 100
                rotate_pix = rotate_angle / angle_pix_rate
                print(f'rotate_angle={rotate_angle}')

                # 转起来!
                action_param = JSwipe(
                    begin=(self.rotate_start_x, self.map_center_y, 0, 0),
                    duration=[500],
                    end=[(self.rotate_start_x + rotate_pix, self.map_center_y, 0, 0)],
                )
                context.run_action_direct(JActionType.Swipe, action_param)
                # 计算到目标点的像素距离
                distance = math.sqrt((target_x - relative_x) ** 2 + (target_y - relative_y) ** 2)
                # 计算需要移动的时间
                time_pix_rate = 1 / 20
                move_time = int(distance * time_pix_rate * 1000)
                print(f"distance={distance}")
                print(f'move_time={move_time}')

                # 走起!
                action_param = JLongPress(
                    target=(self.w_x, self.w_y, 0, 0),
                    duration=move_time,
                )
                context.run_action_direct(JActionType.LongPress, action_param)

        return True

    def _open_map(self, is_DJH: bool,map_x_move_offset=0,map_y_move_offset=0):
        if is_DJH:
            self.context.run_task("帝江号上打开地图")
        else:
            self.context.run_task("打开地图")
        # 放大地图
        self.context.run_action("放大地图")

        # 滑动移动地图使箭头在合适的位置
        if map_x_move_offset != 0:
            swipe_param = JSwipe(
                begin=(self.map_center_x, self.map_center_y, 0, 0),  # 覆盖 JTarget (假设 JTarget 接受元组)
                duration=[500],  # 注意这里是 List[int]
                end=[(self.map_center_x + map_x_move_offset, self.map_center_y, 0, 0)],  # 注意这里是 List[JTarget]
            )
            self.context.run_action_direct(JActionType.Swipe, swipe_param)
        if map_y_move_offset != 0:
            if map_y_move_offset > 0:
                map_start_y = 10
            else:
                map_start_y = 600

            swipe_param = JSwipe(
                begin=(10, map_start_y, 0, 0),  # 覆盖 JTarget (假设 JTarget 接受元组)
                duration=[500],  # 注意这里是 List[int]
                end=[(10, map_start_y + map_y_move_offset, 0, 0)],  # 注意这里是 List[JTarget]
            )
            self.context.run_action_direct(JActionType.Swipe, swipe_param)
