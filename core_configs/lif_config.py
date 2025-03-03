from hang_zhou_spaic.SPAIC import spaic
# from hang_zhou_spaic.SPAIC.spaic
import torch
import math
from hang_zhou_spaic.SPAIC.spaic.Learning.STCA_Learner import STCA
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from enum import Enum
from darwin3_deployment.core_config.core_config import CoreConfig
from darwin3_deployment.ir.dinst import * 
# from darwin3_deployment.ir.dinst import ADD as _ADD

"""
class LIFModel(NeuronModel):
    
    LIF model:
    # V(t) = tuaM * V^n[t-1] + Isyn[t]   # tauM: constant membrane time (tauM=RmCm) # tag
    O^n[t] = spike_func(V^n[t-1])
    

    def __init__(self, **kwargs):
        super(LIFModel, self).__init__()
        # initial value for state variables
        self._variables['V'] = 0.0
        self._variables['O'] = 0.0
        self._variables['Isyn'] = 0.0


        self._constant_variables['Vth'] = kwargs.get('v_th', 1.0)
        self._constant_variables['Vreset'] = kwargs.get('v_reset', 0.0)

        self._tau_variables['tauM'] = kwargs.get('tau_m', 8.0)

        self._operations.append(('Vtemp', 'var_linear', 'tauM', 'V', 'Isyn[updated]'))
        self._operations.append(('O', 'threshold', 'Vtemp', 'Vth'))
        self._operations.append(('V', 'reset', 'Vtemp',  'O[updated]'))

NeuronModel.register("lif", LIFModel)

"""


register = {
    "vt":0,
    "vth":1,
    "i":2,
    'rpd':3,
    'wgtsum':4,
    'res':5,
    'rand':6,
    'wgt':7,
    'pra01':0xD,
    'pra23':0xE,
    'pra45':0xF
}
""" for i in range(len(register)):
    register.i =  """


def spaic_lif_config(vth, tauM=3/4):
    # 输入量化后的阈值， 衰减固定是15/16 在spaic 中需要 修改lif 的 tau_m 为 tau_m=(-0.1 / np.log(3 / 4)) 因为 spaic 自带一层 np.exp(-dt / tau)
    # 用于实现spaic的LIF 神经元
    """
    vt = vt * tauM
    vt = vt + wgsum
    O = func(vt)
    
    vt = vt + wgtsum if FCS == 1
    vt = p1 + vt x p0 + wgtsum x c6 + i x c7 if FCS ==0 其中 c6 = CPAR6 c7 = CPAR7
    """
    """ inference_state_settings={
        "vth": "vth",
    }, """


    core_config = CoreConfig(

        assembly_program=[
            LSIS(ls=LSIS.LS.LOAD, nph=0b0100_0100), # vt weightsum
            
            SUB(rs=3, rt=3, ns=0), # r3 = 0
            ADD(rs=3, rt=0, ns=0), # r3 = vt
            
            SFTI(rs=3, sf=SFTI.SF.LEFT, ns=0, imme=1), # r3 << 1: r3 = 2vt
            ADD(rs=0, rt=3, ns=0), # vt' = vt + 2vt = 3vt

            SFTI(rs=3, sf=SFTI.SF.LEFT, ns=0, imme=1), # r3 << 1: r3 = 4vt
            ADD(rs=0, rt=3, ns=0), # vt' = 3vt + 4vt = 7vt

            SFTI(rs=3, sf=SFTI.SF.LEFT, ns=0, imme=1), # r3 << 1: r3 = 8vt
            ADD(rs=0, rt=3, ns=0), # vt' = 7vt + 8vt = 15vt

            SFTI(rs=0, sf=SFTI.SF.RIGHT, ns=0, imme=4), # 右移4位 vt' >> 4 : vt' = 15vt / 16 
            
            UPTVT(fcs=1, fcph=0), # 更新电压 vt += wgtsum
            GSPRS(rs=0, gspm=GSPRS.GSPM.DEFAULT, rstn=0, gsp=1, vcp=1, rsm=GSPRS.RSM.ZERO), # 复位:是 发放脉冲：是 与阈值比较：是 重置到：0
            LSIS(ls=LSIS.LS.STORE, nph=0b0100_0000), # 把膜电压存回去 
            NPC(),
        ]
    )
    
    core_config.set_register("R1", int(vth)) # 这么写和 inference_settings 一样的吗
    return core_config