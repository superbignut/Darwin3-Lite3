"""
    测试麦克风输入

"""

# -*- coding: utf-8 -*-
"""
可以成功录音

"""

import pyaudio
import wave

# 配置录音参数
FORMAT = pyaudio.paInt16
CHANNELS = 1 # 具体的参数是通过test文件查出来的
RATE = 16000
CHUNK = 1024 * 8
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()

# 打开音频流
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK, input_device_index=27)

print("录音中...")

frames = []

# 录制音频
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束")

# 停止和关闭音频流
stream.stop_stream()
stream.close()
audio.terminate()

# 将录制的音频保存为wav文件
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()