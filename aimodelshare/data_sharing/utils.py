import stat # needed for file stat
import os

# arguments: the function that failed, the path 
# it failed on, and the error that occurred.
def redo_with_write(redo_func, path, err):
    os.chmod(path, stat.S_IWRITE)
    redo_func(path)