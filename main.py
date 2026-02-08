from maa.tasker import Tasker
from maa.toolkit import Toolkit
from maa.controller import (
    AdbController
)

from agent.config import set_resource_path
from agent.register_for_local import *


def main():
    user_path = "./"
    Toolkit.init_option(user_path)
    set_resource_path(is_remote=False)
    # If not found, try running as administrator
    # for 如果是蓝叠模拟器，需要指定adb路径
    # adb_devices = Toolkit.find_adb_devices(r"C:\Users\leon\Downloads\platform-tools-latest-windows\platform-tools\adb.exe")
    # for mumu12 模拟器
    adb_devices = Toolkit.find_adb_devices()
    if not adb_devices:
        print("No ADB device found.")
        exit()

    device = adb_devices[0]
    print(f'device={device}')
    controller = AdbController(
        adb_path=device.adb_path,
        address=device.address,
        screencap_methods=device.screencap_methods,
        input_methods=device.input_methods,
        config=device.config,
    )

    controller.post_connection().wait()

    # resource
    # Load  for do recognition and action
    resource_path = "./assets/resource"
    res_job = resource.post_bundle(resource_path)
    res_job.wait()

    # task
    tasker = Tasker()
    tasker.bind(resource, controller)
    if not tasker.inited:
        print("Failed to init MAA.")
        exit()
    # 起点请在四号谷底区域。
    task_entry = [
        "打开app",
        # "协议空间-打开地图",
        # "制作一个装备",
        # "进行1次简易制作",
        # "提升一次武器等级",
        # "日常活跃度领取",
        # "枢纽区基地电站猫头鹰",
        # "工人之家猫头鹰",
        # "源石研究所猫头鹰",
        # "谷地通道猫头鹰",
        # "矿脉源区医疗站上猫头鹰",
        # "矿脉源区医疗站下猫头鹰",
        # "总控中枢"
    ]
    pipeline_override = {
        "协议空间-一级选择": {
            "recognition": {
                "param": {
                    "expected": [
                        "干员养成"
                    ],
                },
            }
        },
        "协议空间-二级选择1": {
            "recognition": {
                "param": {
                    "expected": [
                        "协议空间·干员进阶"
                    ],
                },
            }
        },
        "协议空间-二级选择2": {
            "recognition": {
                "param": {
                    "expected": [
                        "协议空间·干员进阶"
                    ],
                },
            }
        },
        "协议空间-等级选择": {
            "recognition": {
                "param": {
                    "expected": [
                        "三级"
                    ],
                },
            }
        }
    }
    for entry in task_entry:
        task_detail = tasker.post_task(entry, pipeline_override).wait().get()
        print(f'task_detail={task_detail}')
        print(f"{entry}执行完成")

if __name__ == "__main__":
    main()
