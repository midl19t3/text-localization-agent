import os


def ensure_folder(dir_path, exist_ok=True):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=exist_ok)
