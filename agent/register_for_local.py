import inspect
from agent import custom_action
from agent import custom_reco
from maa.resource import Resource
from maa.custom_action import CustomAction
from maa.custom_recognition import CustomRecognition

resource = Resource()

# 遍历 custom_action 模块中的所有成员
for name, obj in inspect.getmembers(custom_action):
    # 筛选条件：是类 + 继承自 CustomAction + 不是 CustomAction 基类本身
    if (inspect.isclass(obj) and
            issubclass(obj, CustomAction) and
            obj is not CustomAction):
        # 实例化对象并注册
        # 注册名默认使用类名，你也可以根据需要修改
        # 优先级：自定义属性 > 类名
        # 使用 getattr 获取 ACTION_NAME，如果没有则使用原来的 name (类名)
        display_name = getattr(obj, "ACTION_NAME", name)
        resource.register_custom_action(display_name, obj())
        print(f"Successfully registered custom action: {display_name}")

# 遍历 custom_reco 模块中的所有成员
for name, obj in inspect.getmembers(custom_reco):
    # 筛选条件：是类 + 继承自 CustomRecognition + 不是 CustomRecognition 基类本身
    if (inspect.isclass(obj) and
            issubclass(obj, CustomRecognition) and
            obj is not CustomRecognition):
        # 实例化对象并注册
        # 注册名默认使用类名，你也可以根据需要修改
        # 使用 getattr 获取 ACTION_NAME，如果没有则使用原来的 name (类名)
        display_name = getattr(obj, "RECO_NAME", name)
        resource.register_custom_recognition(display_name, obj())
        print(f"Successfully registered custom recognition: {display_name}")
