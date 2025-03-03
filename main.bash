# 开始之前先手动 ssh 连一下 darwin， 如果能连上的话再 sudo bash main.bash

echo "Gouzi start..."

set -e

# 配置 darwin ip 地址
ifconfig usb1 172.31.111.31/24

echo "Connecting darwin3 runtime..."

ssh root@172.31.111.35 "darwin3_runtime_server.py &" ||  { echo "SSH command failed"; exit 1; }


sleep 2

(
    # 设置最外层搜索路径后，启动主脚本
    # 之所以放在API内是darwin工具链的限制
    PYTHONPATH=. python3 ./API_4.0/apps/model/main_darwin.py || { echo "Python script failed"; exit 1; }

)