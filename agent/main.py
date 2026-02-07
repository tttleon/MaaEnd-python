import sys
import os

sys.path.insert(0, os.getcwd())

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit
from agent.register_for_remote import *
from agent.config import *


def main():
    Toolkit.init_option("./")
    set_resource_path(is_remote=True)

    if len(sys.argv) < 2:
        print("Usage: python main.py <socket_id>")
        print("socket_id is provided by AgentIdentifier.")
        sys.exit(1)
        
    socket_id = sys.argv[-1]

    print(f"start up agent server with socket id {socket_id}")
    AgentServer.start_up(socket_id)
    AgentServer.join()
    print(f"agent server with socket id {socket_id} joined")
    AgentServer.shut_down()
    print(f"agent server with socket id {socket_id} shut down")



if __name__ == "__main__":
    # 如果没有提供socket_id，默认使用0eeecfdf-acbc-4973-b934-98e2f5d49baa,给本地调试器连接使用
    if len(sys.argv) < 2:
        id ='0eeecfdf-acbc-4973-b934-98e2f5d49baa'
        sys.argv.append(id)
    main()
