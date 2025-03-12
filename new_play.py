import os
 
def play_wav_with_aplay(file_path):
    os.system(f'aplay "{file_path}"')
 
# 使用示例
play_wav_with_aplay('wang_wang.wav')