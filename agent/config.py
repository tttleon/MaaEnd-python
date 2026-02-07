resource_path_local = "assets/resource"
resource_path_remote = "../resource"
resource_path = ""

def set_resource_path(is_remote: bool):
    global resource_path
    resource_path = resource_path_remote if is_remote else resource_path_local
def get_resource_path() -> str:
    return resource_path
