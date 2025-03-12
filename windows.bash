# 这个是在windows 上跑的 bash
#  当然 这是在模拟器中

echo "Gouzi start..."

set -e


sleep 2

(
    echo "Starting main server..."
    PYTHONPATH=. "C:/Program Download/Python-complier/python.exe" ./API_4.0/apps/model/main_darwin.py
    # 加上 ysc 还是因为同样的 root + pyaudio 语音会报错
    # sleep 22
)

exit 0

# 启动 imu 客户端
(
    echo "Starting imu client..."
    export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages:$PYTHONPATH
    python2 ./hang_zhou_client/client_imu.py &
)

# exit 0

# 启动 vedio 客户端
(
    echo "Starting color client..."
    cd client_video
    /home/ysc/.local/bin/pipenv run python client_video.py & 
)

# 启动 语音 客户端
(
    
    echo "Starting dmx client..."
    # 由于dmx 的环境 在 ../show_file 中配好了，就不重新配了
    cd ../show_file/                
    # 这里 pyaudio 报权限的错误，好像是不能用 sudo 来运行，就只好切回用户 ysc
    sudo -u ysc bash -c '
        export http_proxy=http://192.168.1.100:7890
        export https_proxy=http://192.168.1.100:7890
        /home/ysc/.local/bin/pipenv run python client_dmx.py &'
)

# 启动 酒精检测、电池电量客户端，这两个手动启动就ok

echo "Bash run successfully..."

