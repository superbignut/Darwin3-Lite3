# 这个是在windows 上跑的 bash
# 这样的话， 对比如大模型 和 摄像头的测试 就不用到狗上去，
# 但是狗的交互的动作的体现还是要去调试

echo "Gouzi start..."

set -e

export MY_FLAG

MY_FLAG="dev" # 传递给 python 用于判定是不是 开发模式

sleep 2

(


    echo "Starting main server..."
    PYTHONPATH=. "C:/Program Download/Python-complier/python.exe" ./API_4.0/apps/model/main_darwin.py &
    # 加上 ysc 还是因为同样的 root + pyaudio 语音会报错
    sleep 10
)

# exit 0

# 启动 imu 客户端
if false; then
    (
        echo "Starting imu client..."
        export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages:$PYTHONPATH
        python2 ./hang_zhou_client/client_imu.py &
    )
fi

# exit 0

# 启动 vedio 客户端
(
    echo "Starting color client..."
    cd client_video
    "C:/Program Download/Python-complier/python.exe" client_video.py &
)

# exit 0
# 启动 语音 客户端
(
    
    echo "Starting dmx client..."
    # 由于dmx 的环境 在 ../show_file 中配好了，就不重新配了
    cd show_file/                
    # 这里 pyaudio 报权限的错误，好像是不能用 sudo 来运行，就只好切回用户 ysc


        "C:\conda\envs\quanto\python.exe" client_dmx.py &
)

# 启动 酒精检测、电池电量客户端，这两个手动启动就ok

echo "Bash run successfully..."

