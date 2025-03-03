from hang_zhou_spaic.SPAIC import spaic
# from hang_zhou_spaic.SPAIC.spaic
import torch
import math
from hang_zhou_spaic.SPAIC.spaic.Learning.STCA_Learner import STCA
from hang_zhou_spaic.SPAIC.spaic.Learning.Learner import Learner
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum
import os


def single_test():
    device = torch.device("cuda:0")

    input_node_num_origin = 16

    input_num_mul_index = 16 #  把输入维度放大16倍

    assign_label_len = 10

    input_node_num = input_node_num_origin * input_num_mul_index #  把输入维度放大10倍

    output_node_num = 3 # 这里不写成4 ， 如果输入全是 0 的话， 就不用传播了

    label_num = 100 # 这里要不了这么多

    bat_size = 1

    backend = spaic.Torch_Backend(device)
    backend.dt = 0.1

    run_time = 256 * backend.dt  # runtime 能不能小一点呢， 也需要测试一下吧

    time_step = int(run_time / backend.dt)
    class YscNet(spaic.Network):
        def __init__(self):
            super().__init__()

            self.input = spaic.Encoder(num=input_node_num, time=run_time, coding_method='poisson', unit_conversion=0.8) # 就是给发放速率乘了因子,from 论文

            self.layer1 = spaic.NeuronGroup(label_num, model='lifstdp_ex') # stdp_ex 比 stdp_ih 多了一层阈值的衰减 \tao_\theta, 论文中有提到这个自适应的阈值
            
            self.layer2 = spaic.NeuronGroup(label_num, model='lifstdp_ih') # 维度都是100

            self.output = spaic.Decoder(num=label_num, dec_target=self.layer1, time=run_time, coding_method='spike_counts') # layer1作为了输出层, 兴奋次数作为输出

            self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full', weight=np.random.rand(label_num, input_node_num) * 0.3) # 100 * 784 # 这里其实可以给的高一点， 反正会抑制下去
            
            self.connection2 = spaic.Connection(self.layer1, self.layer2, link_type='full', weight=np.diag(np.ones(label_num)) * 22.5 ) # 这里
            
            self.connection3 = spaic.Connection(self.layer2, self.layer1, link_type='full', weight=( np.ones((label_num, label_num)) - np.diag(np.ones(label_num)) ) * (-120)) # 起到一个抑制的作用，除了1-1的前一个，侧向抑制，并起到竞争的效果

            self._learner = Learner(algorithm='nearest_online_stdp', trainable=self.connection1, run_time=run_time) # 这里也只是训练 从输入到第一层的连接，其余层不变

            self.reward = spaic.Reward(num=label_num, dec_target=self.layer1, coding_time=run_time, coding_method='environment_reward', dec_sample_step=1) # 采样频率是每个时间步一次
            #
            self.mon_weight = spaic.StateMonitor(self.connection1, 'weight', nbatch=-1)
            
            self.set_backend(backend)

            self.buffer = [[] for _ in range(output_node_num)] # 0, 1 投票神经元的buffer # 这里不能写成 [[]] * 4 的形式， 否则会出问题

            self.assign_label = None # 统计结束的100 个神经元的代表的情感对象是什么


        def step(self, data, reward=1): # reward 要想加进去的话, 需要去修改一下stdp 的算法

            self.input(data) # 输入数据

            self.reward(reward) # 这里1，rstdp 退化为stdp 输入奖励 0 则不更新权重

            self.run(run_time) # 前向传播

            return self.output.predict # 输出 结果
        
    Net = YscNet()
    Net.reward(1)
    Net.state_from_dict(filename=r"hang_zhou_spaic\save_600\real_ysc_model_mic", device=device)
    # Net = spaic.Network_loader.network_load(filename='save_lif_200',device=device)
    for i in range(10):
        data =  torch.tensor([2 if k < 20 else 0 for k in range(input_node_num)], device=device).unsqueeze(0)

        output = Net.step(data=data)
         
        output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)


        data =  torch.tensor([2 if k > 20 else 0 for k in range(input_node_num)], device=device).unsqueeze(0)
        output = Net.step(data=data)





def quant():
    class Symmetric_Quantize:
        class Target_Precision(Enum):
            INT8=127 #  #  这里暂时先用这个试一试
            INT16=32767
    
        def __init__(self, precision):
            self.upper_bound = precision.value
            self.lower_bound = - precision.value
            self._scalar_factor = None
            self._bias = None
            self.q = 0.97
        
        def _get_scale_factor_score_weight(self, q, weight): # 这里我的权重本来就是 2维的，flatten 之后变为 1 维
            # print(weight)

            topq = torch.abs(weight).flatten(1).quantile(dim=1, q=q) # 这里的quantile 其实是找到一个分位数的概念
            _scalar_factor = self.upper_bound / topq # 计算放大因数
            # print("scalar_factor is :", _scalar_factor)

            _new_weight = torch.clamp(torch.round(_scalar_factor[:, None] * weight), min=self.lower_bound, max=self.upper_bound)
            # print(_new_weight)

            _quant_back = _new_weight / _scalar_factor[:, None]
            _quant_loss = torch.nn.functional.mse_loss(_quant_back, weight)

            return _quant_loss, _scalar_factor, _new_weight # 返回量化损失，缩放因子，缩放后的权重
            
        def _get_scale_factor_score_weight_unit(self, q, weight):
            # 这里相对于原来的量化方式 ，这里的应该是 统一量化， 之前的应该是 分神经元 量化

            topq = torch.abs(weight).flatten(1).quantile(dim=1, q=q) 
            
            _scalar_factor = torch.sum(self.upper_bound / topq) / len(topq) # 求了量化因子 的平均值
            # print("scalar_factor is :", _scalar_factor)

            _new_weight = torch.clamp(torch.round(_scalar_factor * weight), min=self.lower_bound, max=self.upper_bound)
            # print(_new_weight)

            _quant_back = _new_weight / _scalar_factor# [:, None]
            _quant_loss = torch.nn.functional.mse_loss(_quant_back, weight)

            return _quant_loss, _scalar_factor, _new_weight # 返回量化损失，缩放因子，缩放后的权重
        
        
        def search_reasonable_quant_weight(self, weight):
            # 这里就不选了， 直接用0.97
            _score, _scalar_factor, _new_weight = self._get_scale_factor_score_weight_unit(q=self.q, weight=weight)
            print("origin_weight", weight)
            print("quant_loss is: ", _score)
            print("scalar_factor is: ", _scalar_factor)
            print("_new_weight is: ", _new_weight)
            
            return _scalar_factor, _new_weight

        def __call__(self, origin_weight):
            # print(origin_weight)
           return self.search_reasonable_quant_weight(weight=origin_weight)

    # Net = spaic.Network_loader.network_load(filename='save_lif_200', device=device)

    quant_obj = Symmetric_Quantize(Symmetric_Quantize.Target_Precision.INT8) # 量化权重


    weight_value = torch.load(r'C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\hang_zhou_spaic\save_600\real_ysc_model_mic\parameters\_parameters_dict.pt')

    _scalar_factor, _new_weight= quant_obj(weight_value['autoname1<net>_connection1<con>:autoname1<net>_layer1<neg><-autoname1<net>_input<nod>:{weight}']) # 量化权重
    # print(torch.sum(_scalar_factor), len(_scalar_factor))

    torch.save(_scalar_factor, f="_scalar_factor.pth")
    # average_scalar_factor = torch.sum(_scalar_factor) / len(_scalar_factor)

    # print(_scalar_factor.detach().cpu().numpy())
    
    #  print(np.clip(np.zeros(100, ) - np.round(40 * _scalar_factor.detach().cpu().numpy()), a_min=-32767, a_max=32767))

    # print("average scalar factor is: ", average_scalar_factor) # 这里 使用 平均后 的缩放因子 来 放大达尔文上 的神经元参数 如 阈值、静息电压等

    vth_level = Symmetric_Quantize.Target_Precision.INT8.value

    vth_origin = -52

    v_reset = -60

    decay_v = 0.99

    th_inc = 24 # 0.05 * 490

    # _new_vth = torch.clamp(vth_origin * _scalar_factor, min=-vth_level, max=vth_level).detach().cpu().numpy()[0].item()


    torch.save(_new_weight, f='quant_input_layer1.pth')


    # print("_new_vth is: ", _new_vth)

def delete_dir_file(dir_path, root_dir_rm=False):
    """
    递归删除文件夹下文件和子文件夹里的文件，不会删除空文件夹
    :param dir_path: 文件夹路径
    :return:
    """
    if not os.path.exists(dir_path):
        return
    # 判断是不是一个文件路径，并且存在
    if os.path.isfile(dir_path) and os.path.exists(dir_path):
        os.remove(dir_path)  # 删除单个文件
    else:
        file_list = os.listdir(dir_path)
        for file_name in file_list:
            delete_dir_file(os.path.join(dir_path, file_name), root_dir_rm=True)
    if root_dir_rm == True and os.path.exists(dir_path):
        os.rmdir(dir_path)
    
    
def compile_to_darwin():

    from darwin3_deployment.codegen import dump_input_neuron, add_connections, dump_pop, dump_output_neuron
    from darwin3_deployment.ir.net_population import PhysicalPopulation
    from darwin3_deployment.core_config.get_model_config import get_model_config
    from darwin3_deployment.pops_data import PopsDataConfig
    from darwin3_deployment.codegen import add_standard_connection
    from core_configs.lif_config import spaic_lif_config
    from core_configs.stdpexlif_config_timestep import spaic_stdpexlif_ts_config
    from core_configs.stdpihlif_config_timestep import spaic_stdpihlif_ts_config
    from core_configs.stdpexlif_config_timestep_learn import spaic_stdpexlif_ts_learn_config
    from core_configs.stdpihlif_config_timestep_learn import spaic_stdpihlif_ts_learn_config
    from core_configs.test import test_lif

    time_step = 25

    input_node_num = 256

    label_num = 100

    _scalar_factor = torch.load(f="_scalar_factor.pth")
    
    _scalar_factor = np.sum(_scalar_factor.detach().cpu().numpy()) / 100 / 2 # 这里 会超量程 所以 取了一半 取值 400 

    # print(_scalar_factor)

    input0_neurons = PhysicalPopulation(shape=[input_node_num, ], coord=[-1, 1], pop_position="input") # 使用负坐标， 表示不在darwin上
    layer1_neurons = PhysicalPopulation(shape=[label_num,  ], coord=[ 0, 2]) #
    layer2_neurons = PhysicalPopulation(shape=[label_num,  ], coord=[ 0, 3]) #
    output_neurons = PhysicalPopulation(shape=[label_num,  ], coord=[-1, 3], pop_position='output') # 使用负坐标 

    emo_net_weight = torch.load(r"C:\Users\bignuts\Desktop\ZJU\hang_zhou\alcohol\quant_input_layer1.pth").detach().cpu().numpy() # 加载权重

    """     add_connections('full',   
                    weight=emo_net_weight, 
                    pre_pops=input0_neurons, 
                    post_pops=layer1_neurons) #  """
    

    # 使用 标准连接 代替 全连接 以 支持学习功能
    post_neu_ids = list(range(label_num))       # 标准连接 后神经元个数
    for pre_neu_id in range(input_node_num):    # 标准连接 前神经元 遍历
        add_standard_connection(weight=emo_net_weight[:, pre_neu_id], 
                                pre_pop=input0_neurons, 
                                post_pop=layer1_neurons,
                                pre_neu_id=pre_neu_id, 
                                post_neu_ids=post_neu_ids)
    

    add_connections('full',   
                    weight=np.diag(np.ones(label_num)) * 22 * 5, # 这里的 5  对 下一层连接进行补偿
                    pre_pops=layer1_neurons, 
                    post_pops=layer2_neurons) # 
    add_connections('full',   
                    weight=( np.ones((label_num, label_num)) - np.diag(np.ones(label_num)) ) * (-60 * 2),  #  由于 这里已经无法进一步 放大， 所以对上一层连接 进行放大
                    pre_pops=layer2_neurons, 
                    post_pops=layer1_neurons) # 
    add_connections('output', 
                    weight=None, 
                    pre_pops=layer1_neurons, 
                    post_pops=output_neurons)

    pops_data = {}
    pops_data['layer1'] = {} # layer1 是stdpex 
    # vreset 的 区间是 -32768 ~ 32768 是 16位有符号寄存器
    pops_data['layer1']['core_config'] = spaic_stdpexlif_ts_learn_config(vreset=-60*400, timestep=time_step, th_inc=25, th_sub=1) 
    pops_data['layer1']['vth_theta'] = ( np.zeros((label_num, )) + 0 ) # 所有的vth_theta最开始都是0， 每次脉冲增加 25
    pops_data['layer1']['my_vth'] = ( np.zeros(label_num, ) - 52 * 400 ) # 初始化是-52 # 16位有符号
    pops_data['layer1']['my_loop_index'] = ( np.zeros(label_num, ) ) # 初始化是 0  

    # pops_data['layer1']['1ax'] = ( np.zeros(label_num, ) + 0 ) #   迹初始化是 0 使用寄存器reset 代替


    pops_data['layer2'] = {} # layer2 就是lif 0.9 的衰减
    # vreset 的 区间是 -32768 ~ 32768 是 16位有符号寄存器
    pops_data['layer2']['core_config'] = spaic_stdpihlif_ts_learn_config(vreset=-45, timestep=time_step) #     25    1  
    pops_data['layer2']['my_vth'] = ( np.zeros(label_num, ) - 40) # 初始化是-52 # 16位有符号
    pops_data['layer2']['my_loop_index'] = ( np.zeros(label_num, ) ) # 初始化是 0 

    pops_data_config = PopsDataConfig()
    pops_data_config.set_pops_data(pops_data) # 注册

    out_dir = "./API_4.0/apps/config_files"

    delete_dir_file(out_dir, root_dir_rm=False)    # 清空输出文件夹

    dump_pop(pop_list=layer1_neurons, pop_name='layer1',output_dir=out_dir) # Todo  这个要怎么设定
    dump_pop(pop_list=layer2_neurons, pop_name='layer2',output_dir=out_dir) # Todo  这个要怎么设定
    dump_input_neuron(input_pops=input0_neurons, output_dir=out_dir)
    dump_output_neuron(output_pops=layer1_neurons, pop_name="output_pop", output_dir=out_dir)

def input_func(input_ls, unit_conversion=0.8, dt=0.1):# 这连个参数对标spaic的possion_encoder
    # import numpy as np
    # rand_input = np.where(np.random.rand(*input_ls.shape) < input_ls * unit_conversion * dt)
    a = (np.random.rand(*input_ls.shape) < input_ls * unit_conversion * dt)

    return np.nonzero(a[0].astype(int))[0]
    # temp = torch.rand(self.shape, device=self.device).le(self.source * self.unit_conversion * self.dt).type(self._backend.data_type)
    
ls = np.zeros(100,)
""" def darwin_step():
    global ls
    for _ in range(25):
        input_ls = np.array([[1 if i==2 or i == 5 else 0.01 for i in range(16)] * 16])
        a = input_func(input_ls)
        out = test.run_darwin3_withoutfile(spike_neurons=[a])

        for i in range(len(out[0])):
            index = out[0][i][1]
            ls[index] += 1
    print(ls.nonzero())
    print(ls[ls.nonzero()]) """

if __name__ == "__main__":
    # train()
    # single_test()
    # quant()
    
    compile_to_darwin()


# vt 会不会超量程 ？？？