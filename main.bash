# 开始之前先手动 ssh 连一下 darwin， 如果能连上的话再 sudo bash main.bash

export http_proxy=http://192.168.1.100:7890
export https_proxy=http://192.168.1.100:7890

echo "Gouzi start..."

set -e

# 配置 darwin ip 地址
ifconfig usb1 172.31.111.31/24

echo "Connecting darwin3 runtime..."

echo "123123" | ssh root@172.31.111.35 "darwin3_runtime_server.py &" ||  { echo "SSH command failed"; exit 1; }


sleep 2

(
    # 设置最外层搜索路径后，启动主脚本
    # 之所以放在API内是darwin工具链的限制
    echo "Starting main server..."
    PYTHONPATH=. python3 ./API_4.0/apps/model/main_darwin.py &
    sleep 16
)


# 启动 imu 客户端
(
    echo "Starting imu client..."
    export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages:$PYTHONPATH
    python2 ./hang_zhou_client/client_imu.py &
)

# 启动 vedio 客户端
(
    echo "Starting color client..."
    cd client_video
    /home/ysc/.local/bin/pipenv run python gesture.py & 
)

# 启动 语音客户端


# 启动 酒精检测、电池电量客户端

echo "Bash run successfully..."


# Todo  把脚本分为测试和运行两种状态， 测试条件下，把所有的启动和配置都关闭