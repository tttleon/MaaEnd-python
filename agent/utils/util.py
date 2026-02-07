import math

import cv2
import numpy as np


def get_angle_and_bisector(thresh):  # 新增thresh传参，修复全局变量问题
    # 1. 提取轮廓（用传入的thresh，避免全局变量依赖）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("未检测到轮廓，请检查二值化阈值或图像内容！")
        return False
    # 取最大轮廓（避免小噪点轮廓干扰）
    cnt = max(contours, key=cv2.contourArea)

    # 2. 轮廓简化（道格拉斯普克算法），得到多边形顶点
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    points = np.squeeze(approx).astype(np.float32)

    if len(points) < 3:
        print("轮廓顶点数不足3，无法构成三角形/类三角形！")
        return False

    # 3. 计算所有边的长度+端点，存储(长度, 起点, 终点)
    edges = []
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        edges.append((length, p1, p2))
    # 按边长降序排序，取前两条最长边
    edges_sorted = sorted(edges, key=lambda x: x[0], reverse=True)
    edge1, edge2 = edges_sorted[0], edges_sorted[1]
    # 4. 找两条最长边的公共顶点（夹角顶点）
    def is_same_point(p1, p2, tol=1e-3):
        return math.hypot(p1[0]-p2[0], p1[1]-p2[1]) < tol
    common_p = None
    for p in [edge1[1], edge1[2]]:
        if is_same_point(p, edge2[1]) or is_same_point(p, edge2[2]):
            common_p = p
            break
    if common_p is None:
        print("两条最长边无公共顶点，无法计算夹角！")
        return False

    # 5. 构造两条边的单位向量（从公共顶点出发）
    def get_unit_vector(origin, p):
        vec = p - origin
        vec_len = math.hypot(vec[0], vec[1])
        return vec / vec_len if vec_len > 1e-3 else np.array([0,0])
    p1 = edge1[2] if is_same_point(edge1[1], common_p) else edge1[1]
    p2 = edge2[2] if is_same_point(edge2[1], common_p) else edge2[1]
    vec1 = get_unit_vector(common_p, p1)
    vec2 = get_unit_vector(common_p, p2)


    # 7. 计算角平分线（中间线）的单位向量
    bisector_vec = vec1 + vec2
    bisector_vec = get_unit_vector(np.array([0,0]), bisector_vec)

    # 步骤1：获取角平分线向量的x/y分量（opencv坐标：x向右，y向下）
    dx = bisector_vec[0]  # x轴分量：向右为正，向左为负
    dy = bisector_vec[1]  # y轴分量：向下为正，向上为负
    # 步骤2：计算标准反正切角（math.atan2(dy, dx)）：以x轴正方向为0，逆时针为正（-π~π）
    bisector_rad = math.atan2(dy, dx)
    # 步骤3：转换为「顺时针递增」的角度（0~360°）
    bisector_angle = math.degrees(bisector_rad)  # 先转角度制（-180~180°）
    if bisector_angle < 0:
        bisector_angle += 360  # 负数角度转为正角度（0~360°）
    bisector_angle = 180 + bisector_angle
    if bisector_angle > 360:
        bisector_angle -= 360

    # 返回结果新增「角平分线指向角度bisector_angle」
    return bisector_angle


def get_angle_to_target(self_x, self_y, self_angle, target_x, target_y):
    # 1. 计算目标点相对于自身点的位移
    dx = target_x - self_x
    dy = target_y - self_y

    # 2. 计算目标向量的绝对角度 (弧度)
    # atan2 的参数顺序是 (y, x)
    target_rad = math.atan2(dy, dx)

    # 3. 将弧度转换为角度 (如果你习惯用角度的话)
    target_deg = math.degrees(target_rad)

    # 4. 计算相对角度差
    # 结果为：需要转动的度数
    relative_angle = target_deg - self_angle

    # 5. 标准化到 [-180, 180] 之间
    # 这样可以确保机器人总是往最近的方向转，而不是绕一大圈
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360

    return relative_angle


def check_task_success(res):
    """
    判断 MaaFramework 任务流水线是否真正执行成功
    """
    if not res:
        print("错误：任务未产生任何返回结果")
        return False

    # 1. 检查整体状态码 (假设 status 对象有 is_success 属性或对应枚举)
    # 不同的 Maa 绑定库判断方式略有不同，通常 success 对应值为 0 或特定枚举
    # if res.status != maa.define.Status.Success:
    #     print(f"⚠️ 任务整体状态异常: {res.status}")
    entry = res.entry


    # 2. 检查节点执行链
    if not res.nodes:
        print(f"错误：任务节点列表为空，可能 {entry} 节点配置错误")
        return False

    all_completed = True
    failed_nodes = []

    # for node in res.nodes:
    #     print(f'node==={node}')
    #     # 如果任何一个节点 completed 为 False，则视为流程中断
    #     if not node.completed:
    #         all_completed = False
    #         failed_nodes.append(node.name if node.name else f"UnknownNode(ID:{node.node_id})")
    # 只判断最后一个节点的 completed 状态
    last_node = res.nodes[-1]
    if last_node.completed:
        # print(f"✅ 流程终点【{last_node.name}】执行成功")
        return True
    else:
        print(f"{res.entry}流程在最后一步【{last_node.name}】未完成")
        return False

    # 3. 输出详细结论
    # if all_completed:
    #     # print(f"✅ 任务【{res.entry}】执行成功！共完成 {len(res.nodes)} 个节点。")
    #     return True
    # else:
    #     print(f"任务【{res.entry}】执行失败或中断。")
    #     print(f"   已完成节点数: {sum(1 for n in res.nodes if n.completed)}")
    #     print(f"   卡死/失败位置: {failed_nodes[0]}")  # 通常第一个 False 就是断点
    #     return False


# yolo 640*640  rescale to 1280*720
def restore_box(box, ori_w=1280, ori_h=720, target_size=640):
    x, y, w, h = box

    # 1. 计算缩放比例 (等比缩放通常取最小值)
    r = target_size / max(ori_w, ori_h)

    # 2. 计算黑边偏移量 (Padding)
    # 哪边短，哪边就有 padding
    new_unpad_w = ori_w * r  # 1280 * 0.5 = 640
    new_unpad_h = ori_h * r  # 720 * 0.5 = 360

    dw = (target_size - new_unpad_w) / 2  # 宽度偏移
    dh = (target_size - new_unpad_h) / 2  # 高度偏移 (140)

    # 3. 还原坐标
    # 先减去偏移量，再除以比例
    orig_x = (x - dw) / r
    orig_y = (y - dh) / r
    orig_w = w / r
    orig_h = h / r

    return [orig_x, orig_y, orig_w, orig_h]