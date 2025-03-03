import os
import sys
import numpy as np
import torch
import time
# sys.path.append("../scripts")
from darwin3_runtime_api import darwin3_device

# test = darwin3_device.darwin3_device(app_path='../', step_size=10000, ip=['192.168.1.90']) # 172.31.111.35
board = darwin3_device.darwin3_device(app_path='API_4.0/apps/', step_size=1000_000, ip=['172.31.111.35'], spk_print=True) # 172.31.111.35

time.sleep(1)
board.reset()


time.sleep(1)
board.darwin3_init(333)

time.sleep(2)
board.deploy_config()

time.sleep(1)