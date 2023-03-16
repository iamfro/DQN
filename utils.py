import os
import pandas as pd
from functools import wraps
from tkinter import filedialog
from datetime import datetime
from contextlib import ContextDecorator
import time
import tkinter as tk
import zipfile
import sys


class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'w')
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # 每次写入后刷新到文件中，防止程序意外结束
        self.flush()

    def flush(self):
        self.log.flush()


def backup_code(src_dir, back_path, exclude=None):
    with zipfile.ZipFile(back_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            if exclude:
                dirnames[:] = [dir for dir in dirnames if dir not in exclude]
                filenames[:] = [file for file in filenames if file not in exclude]
            fpath = dirpath.replace(src_dir, '')
            for filename in filenames:
                zipf.write(os.path.join(dirpath, filename),
                           os.path.join(fpath, filename))


def browse_dir(**kwargs):
    root = tk.Tk()
    root.withdraw()
    case_dir = filedialog.askdirectory(**kwargs)
    return case_dir


def browse_path(op=True, **kwargs):
    # 参数和tkinter.filedialog.askopenfilename相同
    # https://docs.python.org/3/library/dialog.html#tkinter.filedialog.askopenfilename
    root = tk.Tk()
    root.withdraw()
    if op:
        case_dir = filedialog.askopenfilename(**kwargs)
    else:
        case_dir = filedialog.asksaveasfilename(**kwargs)
    return case_dir


def list_path(work_dir, prefix='', suffix='', key_word=None,
                is_file=True, is_dir=True, reverse=False):
    import glob
    # 前缀和后缀筛选
    path_list = glob.glob(os.path.join(work_dir, prefix + '*' + suffix))
    # 文件名关键字筛选
    if key_word:
        path_list = [p for p in path_list if key_word in p]
    # 文件和目录类型筛选
    path_list = [p for p in path_list
                 if (is_file & os.path.isfile(p)) | (is_dir & os.path.isdir(p))]
    # 文件排序
    if reverse:
        path_list = sorted(path_list, reverse=reverse)
    return path_list


def get_curr_time():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def set_pd_print_option():
    # 切机信息df打印设置
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)


if __name__ == "__main__":
    pass
