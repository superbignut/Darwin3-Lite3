import os
 
def play_wav_with_aplay(file_path):
    os.system(f'aplay "{file_path}"')
 
# 使用示例
# play_wav_with_aplay('wang_wang.wav')


import wave
import numpy as np
 
def crop_wav(input_file, output_file, duration_ratio=0.3):
    # 打开原始 WAV 文件
    with wave.open(input_file, 'rb') as wav_file:
        # 获取参数
        params = wav_file.getparams()
        channels, sampwidth, framerate, nframes = params[:4]
        
        # 计算要裁剪的帧数
        total_frames = nframes
        crop_frames = int(total_frames * duration_ratio)
        
        # 读取音频数据
        frames = wav_file.readframes(crop_frames)
        
        # 创建新的 WAV 文件
        with wave.open(output_file, 'wb') as new_wav_file:
            # 设置参数
            new_wav_file.setparams(params)
            # 写入裁剪后的音频数据
            new_wav_file.writeframes(frames)
 
    # 使用示例
    input_wav_file = 'woof_sad.wav'  # 输入文件路径
    output_wav_file = 'woof_sad_new.wav'  # 输出文件路径
    crop_wav(input_wav_file, output_wav_file)

