import os
import shutil
import tempfile

def delete_files_from_temp_dir(temp_dir_file_deletion_list):
    temp_dir = tempfile.gettempdir()
    for file_name in temp_dir_file_deletion_list:
        file_path = os.path.join(temp_dir, file_name)
        if(os.path.exists(file_path)):
            os.remove(file_path)

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def make_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)