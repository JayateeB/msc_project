import os


def get_root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_config_path():
    return f"{get_root_path()}/configuration"


def get_config_file_path(file):
    return f"{get_config_path()}/{file}"


def get_input_path():
    return f"{get_root_path()}/input"


def get_input_file_path(file):
    return f"{get_input_path()}/{file}"


def get_output_path():
    return f"{get_root_path()}/output"


def get_output_file_path(file):
    return f"{get_output_path()}/{file}"


def ensure_output_path():
    if not os.path.exists(get_output_path()):
        os.makedirs(get_output_path())
