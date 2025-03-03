"""
    把darwin上的权重数据读出来， 这里主要是 还是通过 get neuron state 的函数，去读树突表， 每次从表的权重地址 读100 个
    sort 是由于 app.log 返回的总是乱序， 因此 先排了一个序

    最终的 256 个权重  在 weight_file_path 中， 我是用的是 8位有符号来表示 所以 大致是 -127 到 127  每个权重占 8位

    weight.txt  中一行 是 6 个 8位 也就是 6个权重
"""

import os 
import re
import torch
import matplotlib.pyplot as plt
import numpy as np

file_path = r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\API_4.0\apps\model\app.log"

sorted_file_path = "app.txt"

addr_pattern = r'addr=0x([0-9a-f]+)'

weight_pattern = r'value=0x[0-9a-f]+([0-9a-f]{4})'

temp_re_pattern = r'addr=(0x[0-9a-f]+)'

temp_weight_pattern = r'value=0x([0-9a-f]+)'

weight_file_path = "weight.txt"

def sort_applog():
    ls = []
    with open(file_path, 'r') as f:
        with open(sorted_file_path, 'w') as new_app_f:

            for index, line in enumerate(f):
                ls.append(line)
                # print(line[81:86])
            ls = sorted(ls, key=lambda x : int(x[81:86], 16))

            new_app_f.write(''.join(ls))
        

def main():


    print("successful")
    ls = []  # ls 由于 存在一点乱序， 需要排序一下
    real_addr = []
    with open(sorted_file_path, 'r') as f:
        # ls = [] # ls 用来存储app.log 读到的权重地址
        
        for index, line in enumerate(f):
            if index > 300:
                break
            addr = re.search(addr_pattern, line)
            weight = re.search(weight_pattern, line)
            # print(addr.group(1), weight.group(1),end=" ")
            # print(hex(int(addr.group(1), 16) + int(weight.group(1), 16)))
            
            # ls.append(hex(int(addr.group(1), 16) + int(weight.group(1), 16)))
            
            ls.append((int(addr.group(1), 16), hex(0x10000 + int(weight.group(1), 16))))
        
        sorted_ls = sorted(ls, key=lambda x: x[0]) # (addr_ori, new_addr) 第一个参数 用作 排序， 第二个才是真的地址
        
        for i in range(len(sorted_ls)-1):
            if i > 260:
                break
            if sorted_ls[i+1][0] - sorted_ls[i][0] != 1:
                print(sorted_ls[i], " ", sorted_ls[i+1])
                print("some error occured.")
                break
            else:
                real_addr.append(sorted_ls[i][1]) # 把 排序后的 真实地址拿出来
    # 使用ls 去寻找 真实的权重
    ls_addr_weight = [] # 地址与权重的对应 list
    num_of_weight_17_lines = 255 # 总共有 256 个 大小为 100 的树突
    num_of_weight_17_lines_cnt = 0


    with open(sorted_file_path, 'r') as f:
        with open(weight_file_path, 'w') as wf:
            find_flag = False
            weight_17_lines = [] #
            for index, line in enumerate(f):

                if find_flag == False:
                    temp_re = re.search(temp_re_pattern, line)

                    if temp_re != None and temp_re.group(1) in real_addr: # 从这行开始的 包括这行在内的 17 行 每行 6个权重，共100个， 最后一行有4个可用
                        # print(temp_re.group(1), "yes")
                        weight_17_lines.append(re.search(temp_weight_pattern, line).group(1))
                        find_flag = True
                        continue
                elif len(weight_17_lines) < 17:
                        weight_17_lines.append(re.search(temp_weight_pattern, line).group(1))
                
                # else:
                    
                if len(weight_17_lines) == 17 :    
                    wf.write('\n'.join(weight_17_lines) + '\n'*2)
                    weight_17_lines.clear()
                    find_flag = False
                    num_of_weight_17_lines_cnt += 1
                elif num_of_weight_17_lines_cnt > num_of_weight_17_lines:
                    print("read and convert is over. weigh data is in weight_file_path")
                    break
                    
                    
                        # print(temp_re.group(1), weight.group(1))
def inner_plot_func(weight):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(weight, cmap='hot_r', vmin=0, vmax=256)
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad=0.05)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_aspect("auto")

    plt.colorbar(im, cax=cax)
    fig.tight_layout()
    plt.show()

w_before_darwin  = None 
w_after_darwin = None 
# 加载量化后的权重
def plot_weight_func():
    global w_before_darwin
    w_before_darwin = torch.load(r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\quant_input_layer1.pth").detach().cpu().numpy()
    w_before_darwin = w_before_darwin.reshape(10, 10, 16, 16).transpose(0, 2, 1, 3).reshape(160, 160)
    inner_plot_func(weight=w_before_darwin)
    
# 加载从darwin上读出来的权重
def darwin_plot_weight_func():
    global w_after_darwin
    ls = []
    temp_ls = []
    hex_data = None
    with open(weight_file_path, 'r') as file:
        hex_data = file.read().split()
    
    

    for index, item in enumerate(hex_data):
        # print(item)
        
        for i in range(len(item) // 2):
            num = int(item[i*2: i*2+2], 16) # 把权重分割
            if num > 127:
                num -= 256
            
            temp_ls.append(num)
            if len(temp_ls) == 102:
                temp_ls.pop(-1)
                temp_ls.pop(-1)
                ls.extend(temp_ls)          # 把权重放到ls中
                temp_ls.clear()        
                
            # print(num)
        
        # print(ls)
    print(len(ls))
    
    ls_new = np.array(ls)
    np.savetxt('new_w.txt', ls_new, fmt='%d', delimiter=',')
    w_after_darwin = ls_new.reshape(256, 100).T.reshape(10, 10, 16, 16).transpose(0, 2, 1, 3).reshape(160, 160)
    
    inner_plot_func(weight=w_after_darwin)
if __name__ == '__main__':
    
    sort_applog() # 把log 排序
    main() # 把权重转换 weight_file_path 文件中

    plot_weight_func() # 画量化 的权重 
    darwin_plot_weight_func() # 画出从darwin 读出来的权重
    print(np.sum((w_after_darwin - w_before_darwin) ** 2))