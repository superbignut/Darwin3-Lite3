"""
    把所有的服务端和客户端的代码全都 kill

"""

import os
import signal
import psutil
 
# 要查找的进程名称列表
process_names = ["client_imu.py","client_video.py", "client_dmx.py",  "main_darwin.py"]
 
# 获取所有正在运行的进程
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        # 获取进程命令行
        cmdline = proc.info['cmdline']
        # 检查进程名称是否在列表中
        if any(name in ' '.join(cmdline) for name in process_names):
            pid = proc.info['pid']
            print(f"Killing process {pid} with command line: {' '.join(cmdline)}")
            # 使用kill -9终止进程
            os.kill(pid, signal.SIGKILL)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass
 
print("All specified processes have been killed.")