import struct
import socket,requests
import os
import glob
import random
import time
import re
import json
from pathlib import Path
import numpy as np
from pyfakefs.fake_filesystem_unittest import Patcher

class Transmitter(object):
    """
    用于建立TCP连接的类, 普通用户无需关注
    """
    def __init__(self):
        self.socket_inst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_inst.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect_lwip(self, ip_address):
        self.socket_inst.connect(ip_address)

    def close(self):
        self.socket_inst.close()
    
    def pack_flit_data(self,data_type,west_data:str | Path | bytes ,east_data:str | Path | bytes,north_data:str | Path | bytes,south_data:str | Path | bytes) -> bytes:
        if isinstance(west_data,str):
            if west_data!='':
                with open(west_data, "rb") as file:
                    west_data = file.read()
            else: west_data=b''

        if isinstance(east_data,str):
            if east_data!='':
                with open(east_data, "rb") as file:
                    east_data = file.read()
            else: east_data=b''
    
        if isinstance(north_data,str):
            if north_data!='':
                with open(north_data, "rb") as file:
                    north_data = file.read()
            else: north_data=b''
    
        if isinstance(south_data,str):
            if south_data!='':
                with open(south_data, "rb") as file:
                    south_data = file.read()
            else: south_data=b''

        rslt=[struct.pack("IIIII", data_type,len(west_data),len(east_data),len(north_data),len(south_data)),
              west_data,east_data,north_data,south_data]
        # print(rslt)
        rslt=b''.join(rslt)
        return rslt
    
    def send_flit_data(self,data:bytes) -> int:
        if len(data) > 2**26:
            print("===<2>=== data is larger than 0.25GB")
            print("===<2>=== send flit length failed")
            return 0
        self.socket_inst.sendall(data)
        return 1

    def send_flit_bin(self, flit_bin_file, data_type):
        """
        发送flit
        """
        with open(flit_bin_file, "rb") as file:
            flit_bin = file.read()
        length = len(flit_bin) >> 2
        if length > 2**26:
            print("===<2>=== %s is larger than 0.25GB" % flit_bin_file)
            print("===<2>=== send flit length failed")
            return 0
        send_bytes = bytearray()
        send_bytes += struct.pack("I", length)
        send_bytes += struct.pack("I", data_type)
        send_bytes += flit_bin
        self.socket_inst.sendall(send_bytes)
        return 1

    def send_flit(self, flit_file, directions=0):
        """
        发送flit
        """
        with open(flit_file, "r") as file:
            flit_list = file.readlines()
        length = len(flit_list)
        if length > 2**26:
            print("===<2>=== %s is larger than 0.25GB" % flit_file)
            print("===<2>=== send flit length failed")
            return 0
        print("===<2>=== send flit length succeed")

        j = 0
        while j < length:
            send_bytes = bytearray()
            send_bytes += struct.pack("I", length)
            send_bytes += struct.pack("I", 0x8000)
            for i in range(j, min(j + 16777216 * 4, length)):
                send_bytes += struct.pack("I", int(flit_list[i % length].strip(), 16))
            self.socket_inst.sendall(send_bytes)
            j = j + 16777216 * 4
            if j <= length:
                reply = self.socket_inst.recv(1024)
                print("%s" % reply)
        return 1


class darwin3_device(object):
    """
    用于和Darwin3开发板进行通信的类
    """

    # 定义flit包长度等的相关常亮
    FLIT_TEXT_LENGTH_BYTE = 8
    FLIT_TEXT_NUM_BYTE = 4
    FLIT_TEXT_LENGTH = FLIT_TEXT_NUM_BYTE * (FLIT_TEXT_LENGTH_BYTE + 1)
    FLIT_BINARY_LENGTH_VALUE = 4
    FLIT_BINARY_NUM_VALUE = 4
    FLIT_BINARY_LENGTH = FLIT_BINARY_NUM_VALUE * FLIT_BINARY_LENGTH_VALUE
    CHIP_RESET = 10
    SET_FREQUENCY = 11
    STATE_FLIT = 0x8000
    DEPLOY_FLIT = 0x7000
    RUN_FLIT = 0x7001

    def __init__(
        self, protocol="TCP", ip=['172.31.111.35'], port=[6000], step_size=25000, app_path="../",
    ):
        """
        
        Args:
            protocol (str):   与 Darwin3 开发板通信使用的协议, 默认 TCP, 可选 LOCAL, 暂不支持其它
            ip (list(str)):   Darwin3 板卡设备 ip 序列, 单芯片开发板最多支持两个 ip
                              默认使用 ip[0] 进行上下位机通信 (暂不支持 ip[1] 的连接)
                              (因为有两张网卡, 以太网接口和type-C接口均可使用)
            port (list(int)): 与 Darwin3 开发板通信使用的端口, 默认为 6000 和 6001
                              其中 port[0] 为和 Darwin3 west 端 DMA 进行通信的端口
                              port[0] 为和 Darwin3 east 端 DMA 进行通信的端口
                              最多支持 4 个端口, 对面 DMA 的四个通道 (目前仅支持 2 个)
            step_size (int):  每个时间步维持的 FPGA 时钟周期数, 对应时长为 10ns * step_size * 2
                              (汇编工具介绍与上位机通信流程中有换算关系，对应run_input.dwnc中最开始的配置)
            app_path (str):   模型文件的存储目录, 存储目录格式如下所示
            .
            ├── apps (name user-defined)
            │   ├── config_files
            │   │   ├── spikes.dwnc (generated by script, users don't need to care)
            │   │   ├── 0-1-config.dwnc
            │   │   ├── 0-1-ax.txt
            │   │   ├── 0-1-de.txt
            │   │   ├── 0-2-config.dwnc
            │   │   ├── 0-2-ax.txt
            │   │   ├── 0-2-de.txt
            │   │   ├── 1-1-config.dwnc
            │   │   ├── 1-1-ax.txt
            │   │   ├── 1-1-de.txt
            │   │   ├── input_neuron.json
            │   │   ├── pop_h_1.json
            │   │   ├── pop_h_2.json
            │   │   ├── output_neuron_xxx.json
            │   │   └── ...
            │   ├── deploy_files (generated by script, users don't need to care)
            │   │   ├── deploy_input.dwnc
            │   │   ├── deploy_flitin.txt
            │   │   └── deploy_flitin.bin
            │   ├── input_files (generated by script, users don't need to care)
            │   │   ├── run_input.dwnc
            │   │   ├── run_flitin.txt
            │   │   └── run_flitin.bin
            │   ├── debug_files (generated by script, users don't need to care)
            │   │   ├── get_neuron_state_flitin.bin
            │   │   ├── get_neuron_state_flitin.txt
            │   │   └── ...
            │   ├── output_files (generated by script, results)
            │   │   ├── recv_run_flit.txt
            │   │   └── recv_run_flit.bin
            │   └── model (other app flies, name user-defined)
            ├── darwin3_runtime_api
            │   ├── darwin3_device.py
            │   └── [your scripts using darwin3_device class](optional)
            ├── script
            │   ├── darwin3_runtime_server.py
            │   ├── restart_dma.sh
            │   └── init.sh
            ├── README.md
            └── setup.py
        """
        self.control_url = f"http://{ip[0]}:6001"
        self.protocol = protocol
        if self.protocol == "LOCAL":
            self.ip = "127.0.0.1"
        else:
            self.ip = ip[0]
        self.port = port
        self.step_size = step_size
        self.app_path = app_path
        self.config_path = app_path + "config_files/"
        # self.neuron_path = app_path + "neuron_files/"
        self.neuron_path = app_path + "config_files/"
        self.deploy_path = app_path + "deploy_files/"
        self.input_path = app_path + "input_files/"
        self.output_path = app_path + "output_files/"
        self.debug_path = app_path + "debug_files/"
        self.config_file_format = "*-*-config.dwnc"
        if not os.path.exists(self.deploy_path):
            os.mkdir(self.deploy_path)
        if not os.path.exists(self.input_path):
            os.mkdir(self.input_path)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if not os.path.exists(self.debug_path):
            os.mkdir(self.debug_path)
        with open(self.neuron_path + "input_neuron.json", "r") as f:
            self.input_neuron = json.load(f)
        self.config_neuron_list = []
        search_paths = glob.glob(self.config_path + self.config_file_format)
        for search_path in search_paths:
            file = os.path.basename(search_path)
            self.config_neuron_list.append(re.findall(r"\d+", file))
        self.deploy_from_east = False
        """
        Attributes:
            protocol (str): 和 Darwin3 的连接方式
            ip (str): 和 Darwin3 进行 TCP 连接的 IP 地址
            port (list(int)): 和 Darwin3 进行 TCP 连接的端口号序列
            step_size (int): 时间步长度
            app_path (str): 存储应用的目录
            config_path (str): 配置文件目录
            neuron_path (str): 神经元信息文件目录 (与配置文件目录相同)
            deploy_path (str): 部署文件目录 (.txt && .bin)
            input_path (str): 输入文件目录 (.txt && .bin)
            output_path (str): 输出文件目录 (.txt && .bin)
            debug_path (str): 查询所有神经元状态信息存储目录 (仅用于调试)
            input_neuron (list): 读取 input_neuron.json 文件, 存储为list
            config_neuron_list (list): 需要进行配置的神经元列表
            deploy_from_east (bool): 判断是否需要从东边进行配置
        """
        return

    @staticmethod
    def onehot2bin(port):
        """
        将 one-hot 编码格式转换为普通二进制格式
        Args
            port (int): 输入的 ont-hot 格式的编码
        Returns:
            输出的二进制格式的编码
        
        """
        if port == 1:
            return 0
        if port == 2:
            return 1
        if port == 4:
            return 2
        if port == 8:
            return 3
        if port == 16:
            return 4
        return 0

    """
    flit_gen <-- gen_flit_by_fn <-- gen_filt_parallel
                                <-- gen_filt
    """
 
    @staticmethod
    def __gen_flit__(item, fin, fbin, direct=0, x_from=-1, y_from=-1, **config_list):
        """
        最里层
        """
        # global last_vc
        # global tick
        # global start_tick
        # global stop_tick
        # global clear_tick
        # global pkg_num
        while config_list.get("config_list") != None:
            config_list = config_list["config_list"]
        if direct == 0:
            config_list["pkg_num"] += 1
        tik = int(item[0])
        cmd = "0xc0000000"
        if item[1] == "cmd":
            cmd = item[2]
        while isinstance(cmd, str):
            cmd = eval(cmd)
        cmd = cmd >> 24
        if (
            tik != config_list["tick"]
            and tik > 0
            and (item[1] != "cmd" or cmd != 0b11011000)
        ):
            cmd = 0b011000
            arg = (tik - config_list["tick"] - 1) & 0xFFFFFF
            cmd_f = 0x3
            if direct == 0:
                l = (cmd_f << 30) + (cmd << 24) + arg
                ss_l = b"%08x\n" % l
                fin.write(ss_l)
                fbin.write(struct.pack("I", l))
            else:
                l = (cmd_f << 30) + (cmd << 24)
                ss_l = b"%08x\n" % l
                for i in range(arg + 1):
                    fin.write(ss_l)
                    fbin.write(struct.pack("I", l))
        if tik > 0:
            config_list["tick"] = tik
        # vc   = int(item[1])
        vc = 0
        if vc == 0:
            vc = config_list["last_vc"] << 1
            if vc > 8:
                vc = 1
        elif vc not in (1, 2, 4, 8):
            vc_list = []
            if vc & 0x1:
                vc_list.append(1)
            if vc & 0x2:
                vc_list.append(2)
            if vc & 0x4:
                vc_list.append(4)
            if vc & 0x8:
                vc_list.append(8)
            vc = random.choice(vc_list)
        config_list["last_vc"] = vc
        op = item[1]
        cmd = "0x80000000"
        if op == "cmd":
            cmd = item[2]
            x = 0
            y = 0
            x_src = 0
            x_dff = 0
            x_sig = 0
            y_src = 0
            y_dff = 0
            y_sig = 0
            if "0xc0000001" in cmd:
                config_list["start_tick"] = tik
            if "0xc0000000" in cmd:
                config_list["stop_tick"] = tik
            if "0xd0000000" in cmd:
                config_list["clear_tick"] = tik
                # if config_list["stop_tick"] == -1 or config_list["clear_tick"] != config_list["stop_tick"]:
                #    print("clear tick must follow stop tick in same step!")
                #    sys.exit(1)
        else:
            x = int(item[2])
            y = int(item[3])
            addr = item[4]
            if len(item) > 5:
                data = item[5]
            else:
                data = 0
            if (
                op == "spike"
                or op == "spike_short"
                or op == "reward"
                or op == "reward_short"
                or op == "write"
                or op == "write_risc"
                or op == "read_ack"
                or op == "read_risc_ack"
                or op == "flow_ack"
                or op == "read"
            ):
                if len(item) > 6:
                    x_from = int(item[6])
                if len(item) > 7:
                    y_from = int(item[7])
            if op == "flow":
                if len(item) > 5:
                    x_from = int(item[5])
                if len(item) > 6:
                    y_from = int(item[6])

            x_src = x_from - x
            if x_src > 0:
                x_sig = 1
            else:
                x_src = -x_src
                x_sig = 0
            if y_from != -1 and x_from != -1:
                if x_src != 0:
                    if x_sig == 1:
                        x_dff = -1 - x
                    else:
                        x_dff = 24 - x
                else:
                    x_dff = 0
            elif x_src >= 1:
                x_dff = x_src - 1
            else:
                x_dff = 0
            if x_src == 16:
                x_src = 15
            if y_from == -1:
                if y < 0:
                    y_sig = 1
                    y_dff = -y - 1
                    y_src = -y
                elif y < 24:
                    y_sig = 0
                    y_dff = 0
                    y_src = 0
                else:
                    y_sig = 0
                    y_dff = y - 24
                    y_src = y - 23
            else:
                y_sig = 0
                y_src = y_from - y
                if y_src > 0:
                    y_sig = 1
                    y_dff = y_from - y
                elif y_src < 0:
                    y_src = -y_src
                    y_dff = y - y_from
                else:
                    y_dff = 0
                if y_src == 16:
                    y_src = 15

        if x_dff > 0:
            if x_sig == 1:
                port = "01000"
            else:
                port = "00010"
        else:
            if y_dff == 0:
                port = "00001"
            elif y_sig == 1:
                port = "00100"
            else:
                port = "10000"
        route_id = y
        if y_from != -1:
            route_id = y_from
        else:
            if y < 0:
                route_id = 0
            elif y > 23:
                route_id = 23

        if op == "read_risc_ack":
            if x_dff == 0:
                x_sig = 1
            if y_dff == 0:
                y_sig = 1

        direct = 2
        cmd_tmp = cmd
        while isinstance(cmd_tmp, str):
            cmd_tmp = eval(cmd_tmp)
        pclass = op
        vcnum = (route_id & 0xF) + (direct << 6)
        vcnum2 = direct << 6
        if direct == 1:
            vcnum2 = direct << 6
        port = eval("0b" + port)
        port = darwin3_device.onehot2bin(port)
        if pclass == "cmd":
            while isinstance(cmd, str):
                cmd = eval(cmd)
            l = cmd
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "write_risc" or pclass == "read_risc_ack":
            while isinstance(addr, str):
                addr = eval(addr)
            while isinstance(data, str):
                data = eval(data)
            if pclass == "read_risc_ack":
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                    + (1 << 0)
                )
            else:
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + (addr << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + ((data & 0xFFFF0000) >> 1)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + ((data & 0xFFFF) << 15)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "write" or pclass == "read_ack":
            while isinstance(addr, str):
                addr = eval(addr)
            while isinstance(data, str):
                data = eval(data)
            if pclass == "read_ack":
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                    + (1 << 0)
                )
            else:
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + (addr << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + ((data & 0xFFFFFF000000) >> 21)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + ((data & 0xFFFFFF) << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "read":
            while isinstance(addr, str):
                addr = eval(addr)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x2 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (addr << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "flow":
            data1 = addr
            while isinstance(data1, str):
                data1 = eval(data1)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x3 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (data1 << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "flow_ack":
            data1 = addr
            while isinstance(data1, str):
                data1 = eval(data1)
            data2 = data
            while isinstance(data2, str):
                data2 = eval(data2)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x7 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + (data1 << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + ((data2 & 0x3FFFFFF8000000) >> 24)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + ((data2 & 0x7FFFFFF) << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "spike":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x0 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "spike_short":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x4 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "reward":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x5 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "reward_short":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x6 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
    
    @staticmethod
    def __gen_flit_east__(
        item, fin, fbin, direct=0, x_from=-1, y_from=-1, **config_list
    ):
        """
        最里层
        """
        # global last_vc
        # global tick
        # global start_tick
        # global stop_tick
        # global clear_tick
        # global pkg_num
        while config_list.get("config_list") != None:
            config_list = config_list["config_list"]
        if direct == 0:
            config_list["pkg_num"] += 1
        tik = int(item[0])
        cmd = "0xc0000000"
        if item[1] == "cmd":
            cmd = item[2]
        while isinstance(cmd, str):
            cmd = eval(cmd)
        cmd = cmd >> 24
        if (
            tik != config_list["tick"]
            and tik > 0
            and (item[1] != "cmd" or cmd != 0b11011000)
        ):
            cmd = 0b011000
            arg = (tik - config_list["tick"] - 1) & 0xFFFFFF
            cmd_f = 0x3
            if direct == 0:
                l = (cmd_f << 30) + (cmd << 24) + arg
                ss_l = b"%08x\n" % l
                fin.write(ss_l)
                fbin.write(struct.pack("I", l))
            else:
                l = (cmd_f << 30) + (cmd << 24)
                ss_l = b"%08x\n" % l
                for i in range(arg + 1):
                    fin.write(ss_l)
                    fbin.write(struct.pack("I", l))
        if tik > 0:
            config_list["tick"] = tik
        # vc   = int(item[1])
        vc = 0
        if vc == 0:
            vc = config_list["last_vc"] << 1
            if vc > 8:
                vc = 1
        elif vc not in (1, 2, 4, 8):
            vc_list = []
            if vc & 0x1:
                vc_list.append(1)
            if vc & 0x2:
                vc_list.append(2)
            if vc & 0x4:
                vc_list.append(4)
            if vc & 0x8:
                vc_list.append(8)
            vc = random.choice(vc_list)
        config_list["last_vc"] = vc
        op = item[1]
        cmd = "0x80000000"
        if op == "cmd":
            cmd = item[2]
            x = 0
            y = 0
            x_src = 0
            x_dff = 0
            x_sig = 0
            y_src = 0
            y_dff = 0
            y_sig = 0
            if "0xc0000001" in cmd:
                config_list["start_tick"] = tik
            if "0xc0000000" in cmd:
                config_list["stop_tick"] = tik
            if "0xd0000000" in cmd:
                config_list["clear_tick"] = tik
                # if config_list["stop_tick"] == -1 or config_list["clear_tick"] != config_list["stop_tick"]:
                #    print("clear tick must follow stop tick in same step!")
                #    sys.exit(1)
        else:
            x = int(item[2])
            y = int(item[3])
            addr = item[4]
            if len(item) > 5:
                data = item[5]
            else:
                data = 0
            if (
                op == "spike"
                or op == "spike_short"
                or op == "reward"
                or op == "reward_short"
                or op == "write"
                or op == "write_risc"
                or op == "read_ack"
                or op == "read_risc_ack"
                or op == "flow_ack"
                or op == "read"
            ):
                if len(item) > 6:
                    x_from = int(item[6])
                if len(item) > 7:
                    y_from = int(item[7])
            if op == "flow":
                if len(item) > 5:
                    x_from = int(item[5])
                if len(item) > 6:
                    y_from = int(item[6])

            x_src = x_from - x
            if x_src > 0:
                x_sig = 1
            else:
                x_src = -x_src
                x_sig = 0
            if y_from != -1 and x_from != -1:
                if x_src != 0:
                    if x_sig == 1:
                        x_dff = -1 - x
                    else:
                        x_dff = 24 - x
                else:
                    x_dff = 0
            elif x_src >= 1:
                x_dff = x_src - 1
            else:
                x_dff = 0
            if x_src == 16:
                x_src = 15
            if y_from == -1:
                if y < 0:
                    y_sig = 1
                    y_dff = -y - 1
                    y_src = -y
                elif y < 24:
                    y_sig = 0
                    y_dff = 0
                    y_src = 0
                else:
                    y_sig = 0
                    y_dff = y - 24
                    y_src = y - 23
            else:
                y_sig = 0
                y_src = y_from - y
                if y_src > 0:
                    y_sig = 1
                    y_dff = -1 - y
                elif y_src < 0:
                    y_src = -y_src
                    y_dff = 24 - y
                else:
                    y_dff = 0
                if y_src == 16:
                    y_src = 15

        if x_dff > 0:
            if x_sig == 1:
                port = "01000"
            else:
                port = "00010"
        else:
            if y_dff == 0:
                port = "00001"
            elif y_sig == 1:
                port = "00100"
            else:
                port = "10000"
        route_id = y
        if y_from != -1:
            route_id = y_from
        else:
            if y < 0:
                route_id = 0
            elif y > 23:
                route_id = 23

        if op == "read_risc_ack":
            if x_dff == 0:
                x_sig = 1
            if y_dff == 0:
                y_sig = 1

        direct = 2
        cmd_tmp = cmd
        while isinstance(cmd_tmp, str):
            cmd_tmp = eval(cmd_tmp)
        pclass = op
        vcnum = (route_id & 0xF) + (direct << 6)
        vcnum2 = direct << 6
        if direct == 1:
            vcnum2 = direct << 6
        port = eval("0b" + port)
        port = darwin3_device.onehot2bin(port)
        if pclass == "cmd":
            while isinstance(cmd, str):
                cmd = eval(cmd)
            l = cmd
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "write_risc" or pclass == "read_risc_ack":
            while isinstance(addr, str):
                addr = eval(addr)
            while isinstance(data, str):
                data = eval(data)
            if pclass == "read_risc_ack":
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                    + (1 << 0)
                )
            else:
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + (addr << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + ((data & 0xFFFF0000) >> 1)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + ((data & 0xFFFF) << 15)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "write" or pclass == "read_ack":
            while isinstance(addr, str):
                addr = eval(addr)
            while isinstance(data, str):
                data = eval(data)
            if pclass == "read_ack":
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                    + (1 << 0)
                )
            else:
                l = (
                    (0x2 << 30)
                    + (route_id << 25)
                    + (0x1 << 22)
                    + (port << 19)
                    + (x_sig << 18)
                    + (x_dff << 14)
                    + (y_sig << 13)
                    + (y_dff << 9)
                    + (x_src << 5)
                    + (y_src << 1)
                )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + (addr << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + ((data & 0xFFFFFF000000) >> 21)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + ((data & 0xFFFFFF) << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "read":
            while isinstance(addr, str):
                addr = eval(addr)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x2 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (addr << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "flow":
            data1 = addr
            while isinstance(data1, str):
                data1 = eval(data1)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x3 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (data1 << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "flow_ack":
            data1 = addr
            while isinstance(data1, str):
                data1 = eval(data1)
            data2 = data
            while isinstance(data2, str):
                data2 = eval(data2)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x7 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + (data1 << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x0 << 30) + ((data2 & 0x3FFFFFF8000000) >> 24)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + ((data2 & 0x7FFFFFF) << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "spike":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x0 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "spike_short":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x4 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "reward":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x5 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
        if pclass == "reward_short":
            dedr_id = addr
            neu_idx = data
            while isinstance(dedr_id, str):
                dedr_id = eval(dedr_id)
            while isinstance(neu_idx, str):
                neu_idx = eval(neu_idx)
            l = (
                (0x2 << 30)
                + (route_id << 25)
                + (0x6 << 22)
                + (port << 19)
                + (x_sig << 18)
                + (x_dff << 14)
                + (y_sig << 13)
                + (y_dff << 9)
                + (x_src << 5)
                + (y_src << 1)
            )
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
            l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack("I", l))
    
    @staticmethod
    def __gen_flit_parallel__(
        x,
        y,
        address,
        value,
        text_buffer,
        text_offset,
        binary_buffer,
        binary_offset,
        **config_list
    ):
        while config_list.get("config_list") != None:
            config_list = config_list["config_list"]
        src_x = x + 1
        if src_x == 16:
            src_x = 15
        # src_y = 0
        diff_x = x
        # diff_y = 0
        # sign_x = 0
        # sign_y = 0

        if diff_x > 0:
            port = 1
        else:
            port = 0
        route_id = y
        # direct   = 2
        # vcnum    = (route_id & 0xf) + (direct << 6)
        # vcnum2   = (direct << 6)
        # msb      = route_id >> 4

        head = (
            (0x2 << 30)
            + (route_id << 25)
            + (0x1 << 22)
            + (port << 19)
            + (diff_x << 14)
            + (src_x << 5)
        )
        body0 = (0x0 << 30) + (address << 3)
        body1 = (0x0 << 30) + ((value & 0xFFFFFF000000) >> 21)
        tail = (0x1 << 30) + ((value & 0xFFFFFF) << 3)
        text_buffer[text_offset : text_offset + darwin3_device.FLIT_TEXT_LENGTH] = (
            b"%08x\n%08x\n%08x\n%08x\n" % (head, body0, body1, tail)
        )
        struct.pack_into("<4I", binary_buffer, binary_offset, head, body0, body1, tail)

    @staticmethod
    def __gen_flit_parallel_east__(
        x,
        y,
        address,
        value,
        text_buffer,
        text_offset,
        binary_buffer,
        binary_offset,
        **config_list
    ):
        while config_list.get("config_list") != None:
            config_list = config_list["config_list"]
        src_x = 24 - x
        # if src_x == 16:
        #     src_x = 15
        # src_y = 0
        diff_x = 23 - x
        # diff_y = 0
        sign_x = 1
        # sign_y = 0

        if diff_x > 0:
            port = 3
        else:
            port = 0
        route_id = y
        # direct   = 2
        # vcnum    = (route_id & 0xf) + (direct << 6)
        # vcnum2   = (direct << 6)
        # msb      = route_id >> 4

        head = (
            (0x2 << 30)
            + (route_id << 25)
            + (0x1 << 22)
            + (port << 19)
            + (sign_x << 18)
            + (diff_x << 14)
            + (src_x << 5)
        )
        body0 = (0x0 << 30) + (address << 3)
        body1 = (0x0 << 30) + ((value & 0xFFFFFF000000) >> 21)
        tail = (0x1 << 30) + ((value & 0xFFFFFF) << 3)
        text_buffer[text_offset : text_offset + darwin3_device.FLIT_TEXT_LENGTH] = (
            b"%08x\n%08x\n%08x\n%08x\n" % (head, body0, body1, tail)
        )
        struct.pack_into("<4I", binary_buffer, binary_offset, head, body0, body1, tail)

    def __gen_flit_by_fn__(self, fn, fin, fbin, direct=0, tc="", **config_list):
        """
        这个函数gen_flit_by_fn负责根据输入文件（通常是.dwnc文件）生成FLIT数据包，
        并将这些数据包写入到文本和二进制文件中。
        FLIT数据包是一种用于神经网络中数据传输的格式，
        它包含了操作类型、目标地址、数据值和其他控制信息。
        函数首先检查输入文件是否存在，如果不存在则直接返回。
        然后，它打开文件并读取所有行，遍历每一行来处理不同的操作。
        对于读操作，它可能需要从其他文件中读取数据并生成对应的FLIT数据包。
        对于写操作，它需要将数据写入FLIT数据包中。
        最后，它使用gen_flit函数生成FLIT数据包，并将这些数据包写入到文本和二进制文件中。
        这个函数通过递归调用自己来处理包含的文件，
        这允许它处理复杂的.dwnc文件，这些文件可能包含对其他文件的引用。
        """
        # print("===========================")
        # print("into gen_flit_by_fn")
        # print("file: " + fn)
        # print("config list before:")
        # print(config_list)
        tc = self.config_path
        while config_list.get("config_list") != None:
            config_list = config_list["config_list"]
        # print("config list after:")
        # print(config_list)
        if not os.path.exists(fn):
            return
        with open(fn, "r", encoding="utf-8") as load_f:
            # print("config list[2]:")
            # print(config_list)
            lines = load_f.readlines()
            for items in lines:
                # print("line: ")
                # print(items)
                # print("config list[3]:")
                # print(config_list)
                item = items.split()
                if len(item) < 2:
                    continue
                if "#" in item[0]:
                    continue
                if item[0] == "<<":
                    self.__gen_flit_by_fn__(
                        self.config_path + item[1],
                        fin,
                        fbin,
                        direct,
                        tc,
                        config_list=config_list,
                    )
                elif item[1] == "read" and len(item) >= 6:
                    tmp = item
                    addr = eval(eval(item[4]))
                    while isinstance(item[5], str):
                        item[5] = eval(item[5])
                    for i in range(int(item[5])):
                        tmp[4] = '"%s"' % hex(addr + i)
                        darwin3_device.__gen_flit__(
                            tmp,
                            fin,
                            fbin,
                            direct,
                            x_from=-1,
                            y_from=-1,
                            config_list=config_list,
                        )
                elif (
                    item[1] == "write"
                    or items[1] == "write_ram"
                    or item[1] == "read_ack"
                    or item[1] == "write_risc"
                    or item[1] == "read_risc_ack"
                ) and len(item) == 5:
                    if item[1] == "write_ram":
                        item[1] = "write"
                    tmp = item
                    tmp.append("")
                    if os.path.exists(tc + item[4]):
                        with open(tc + item[4], "rb") as write_f:
                            tot = int.from_bytes(
                                write_f.read(4), byteorder="little", signed=False
                            )
                            for segment in range(tot):
                                area_id = int(write_f.read(1)[0])
                                t = area_id & 0xF
                                config_word_equal = (t & 0x80) != 0
                                addr = int.from_bytes(
                                    write_f.read(4), byteorder="little", signed=False
                                )
                                length = int.from_bytes(
                                    write_f.read(4), byteorder="little", signed=False
                                )
                                bo = "little"
                                bs = 4
                                di = 1
                                # Dedr
                                if t == 0x01:
                                    addr += 0x10000
                                    bs = 6
                                # Wgtsum
                                if t == 0x02:
                                    addr += 0x4000
                                    bs = 2
                                # Inference State
                                if t == 0x03:
                                    addr += 0x10000
                                    bs = 6
                                    di = -1
                                # Voltage
                                if t == 0x04:
                                    addr += 0x2000
                                    bs = 2
                                # Axon
                                if t == 0x05:
                                    addr += 0x8000
                                    bs = 4
                                # Learn State
                                if t == 0x06:
                                    addr += 0x10000
                                    bs = 6
                                    di = -1
                                # Inst
                                if t == 0x07:
                                    addr += 0x1000
                                    bs = 2
                                # Reg
                                if t == 0x08:
                                    addr += 0x800
                                    bs = 2
                                size = length if config_word_equal else int(length / bs)
                                x, y = int(item[2]), int(item[3])
                                address = np.arange(addr, addr + size * di, di)
                                if config_word_equal:
                                    value = struct.unpack_from(
                                        "<Q", write_f.read(bs) + b"\x00" * (8 - bs)
                                    )[0]
                                    text_buffer = bytes(darwin3_device.FLIT_TEXT_LENGTH)
                                    binary_buffer = bytes(
                                        darwin3_device.FLIT_BINARY_LENGTH
                                    )
                                    darwin3_device.__gen_flit_parallel__(
                                        x,
                                        y,
                                        address,
                                        value,
                                        text_buffer,
                                        0,
                                        binary_buffer,
                                        0,
                                        config_list=config_list,
                                    )
                                    text_buffer = text_buffer * size
                                    binary_buffer = binary_buffer * size
                                else:
                                    buffer = write_f.read(length)
                                    index_byte = np.arange(0, length, bs)
                                    text_buffer = bytearray(
                                        size * darwin3_device.FLIT_TEXT_LENGTH
                                    )
                                    text_offset = np.arange(
                                        0,
                                        size * darwin3_device.FLIT_TEXT_LENGTH,
                                        darwin3_device.FLIT_TEXT_LENGTH,
                                    )
                                    binary_buffer = bytearray(
                                        size * darwin3_device.FLIT_BINARY_LENGTH
                                    )
                                    binary_offset = np.arange(
                                        0,
                                        size * darwin3_device.FLIT_BINARY_LENGTH,
                                        darwin3_device.FLIT_BINARY_LENGTH,
                                    )

                                    def convert(
                                        x,
                                        y,
                                        address,
                                        index_byte,
                                        text_offset,
                                        binary_offset,
                                    ):
                                        buffer_value = buffer[
                                            index_byte : index_byte + bs
                                        ] + b"\x00" * (8 - bs)
                                        value = struct.unpack("<Q", buffer_value)[0]
                                        darwin3_device.__gen_flit_parallel__(
                                            x,
                                            y,
                                            address,
                                            value,
                                            text_buffer,
                                            text_offset,
                                            binary_buffer,
                                            binary_offset,
                                            config_list=config_list,
                                        )

                                    np.frompyfunc(convert, 6, 0)(
                                        x,
                                        y,
                                        address,
                                        index_byte,
                                        text_offset,
                                        binary_offset,
                                    )
                                fin.write(text_buffer)
                                fbin.write(binary_buffer)
                elif (
                    item[1] == "write"
                    or item[1] == "write_ram"
                    or item[1] == "read_ack"
                    or item[1] == "write_risc"
                    or item[1] == "read_risc_ack"
                ) and os.path.exists(tc + item[5]):
                    if item[1] == "write_ram":
                        item[1] = "write"
                    tmp = item
                    x, y = int(item[2]), int(item[3])
                    addr = item[4]
                    while isinstance(addr, str):
                        addr = eval(addr)
                    with open(tc + item[5], "r") as write_f:
                        wlines = write_f.readlines()
                        wlength = len(wlines)
                        address = np.arange(addr, addr + wlength)
                        text_buffer = bytearray(
                            wlength * darwin3_device.FLIT_TEXT_LENGTH
                        )
                        text_offset = np.arange(
                            0,
                            wlength * darwin3_device.FLIT_TEXT_LENGTH,
                            darwin3_device.FLIT_TEXT_LENGTH,
                        )
                        binary_buffer = bytearray(
                            wlength * darwin3_device.FLIT_BINARY_LENGTH
                        )
                        binary_offset = np.arange(
                            0,
                            wlength * darwin3_device.FLIT_BINARY_LENGTH,
                            darwin3_device.FLIT_BINARY_LENGTH,
                        )

                        def convert(x, y, address, line, text_offset, binary_offset):
                            return darwin3_device.__gen_flit_parallel__(
                                x,
                                y,
                                address,
                                int(line, 16),
                                text_buffer,
                                text_offset,
                                binary_buffer,
                                binary_offset,
                                config_list=config_list,
                            )

                        np.frompyfunc(convert, 6, 0)(
                            x, y, address, wlines, text_offset, binary_offset
                        )
                        fin.write(text_buffer)
                        fbin.write(binary_buffer)
                else:
                    darwin3_device.__gen_flit__(
                        item,
                        fin,
                        fbin,
                        direct,
                        x_from=-1,
                        y_from=-1,
                        config_list=config_list,
                    )
        load_f.close()

    def __gen_flit_by_fn_east__(self, fn, fin, fbin, direct=0, tc="", **config_list):
        """
        这个函数gen_flit_by_fn负责根据输入文件（通常是.dwnc文件）生成FLIT数据包，
        并将这些数据包写入到文本和二进制文件中。
        FLIT数据包是一种用于神经网络中数据传输的格式，
        它包含了操作类型、目标地址、数据值和其他控制信息。
        函数首先检查输入文件是否存在，如果不存在则直接返回。
        然后，它打开文件并读取所有行，遍历每一行来处理不同的操作。
        对于读操作，它可能需要从其他文件中读取数据并生成对应的FLIT数据包。
        对于写操作，它需要将数据写入FLIT数据包中。
        最后，它使用gen_flit函数生成FLIT数据包，并将这些数据包写入到文本和二进制文件中。
        这个函数通过递归调用自己来处理包含的文件，
        这允许它处理复杂的.dwnc文件，这些文件可能包含对其他文件的引用。
        """
        # print("===========================")
        # print("into gen_flit_by_fn")
        # print("file: " + fn)
        # print("config list before:")
        # print(config_list)
        tc = self.config_path
        while config_list.get("config_list") != None:
            config_list = config_list["config_list"]
        # print("config list after:")
        # print(config_list)
        if not os.path.exists(fn):
            return
        with open(fn, "r", encoding="utf-8") as load_f:
            # print("config list[2]:")
            # print(config_list)
            lines = load_f.readlines()
            for items in lines:
                # print("line: ")
                # print(items)
                # print("config list[3]:")
                # print(config_list)
                item = items.split()
                if len(item) < 2:
                    continue
                if "#" in item[0]:
                    continue
                if item[0] == "<<":
                    self.__gen_flit_by_fn_east__(
                        self.config_path + item[1],
                        fin,
                        fbin,
                        direct,
                        tc,
                        config_list=config_list,
                    )
                elif item[1] == "read" and len(item) >= 6:
                    tmp = item
                    addr = eval(eval(item[4]))
                    while isinstance(item[5], str):
                        item[5] = eval(item[5])
                    for i in range(int(item[5])):
                        tmp[4] = '"%s"' % hex(addr + i)
                        darwin3_device.__gen_flit_east__(
                            tmp,
                            fin,
                            fbin,
                            direct,
                            x_from=24,
                            y_from=-1,
                            config_list=config_list,
                        )
                elif (
                    item[1] == "write"
                    or items[1] == "write_ram"
                    or item[1] == "read_ack"
                    or item[1] == "write_risc"
                    or item[1] == "read_risc_ack"
                ) and len(item) == 5:
                    if item[1] == "write_ram":
                        item[1] = "write"
                    tmp = item
                    tmp.append("")
                    if os.path.exists(tc + item[4]):
                        with open(tc + item[4], "rb") as write_f:
                            tot = int.from_bytes(
                                write_f.read(4), byteorder="little", signed=False
                            )
                            for segment in range(tot):
                                area_id = int(write_f.read(1)[0])
                                t = area_id & 0xF
                                config_word_equal = (t & 0x80) != 0
                                addr = int.from_bytes(
                                    write_f.read(4), byteorder="little", signed=False
                                )
                                length = int.from_bytes(
                                    write_f.read(4), byteorder="little", signed=False
                                )
                                bo = "little"
                                bs = 4
                                di = 1
                                # Dedr
                                if t == 0x01:
                                    addr += 0x10000
                                    bs = 6
                                # Wgtsum
                                if t == 0x02:
                                    addr += 0x4000
                                    bs = 2
                                # Inference State
                                if t == 0x03:
                                    addr += 0x10000
                                    bs = 6
                                    di = -1
                                # Voltage
                                if t == 0x04:
                                    addr += 0x2000
                                    bs = 2
                                # Axon
                                if t == 0x05:
                                    addr += 0x8000
                                    bs = 4
                                # Learn State
                                if t == 0x06:
                                    addr += 0x10000
                                    bs = 6
                                    di = -1
                                # Inst
                                if t == 0x07:
                                    addr += 0x1000
                                    bs = 2
                                # Reg
                                if t == 0x08:
                                    addr += 0x800
                                    bs = 2
                                size = length if config_word_equal else int(length / bs)
                                x, y = int(item[2]), int(item[3])
                                address = np.arange(addr, addr + size * di, di)
                                if config_word_equal:
                                    value = struct.unpack_from(
                                        "<Q", write_f.read(bs) + b"\x00" * (8 - bs)
                                    )[0]
                                    text_buffer = bytes(darwin3_device.FLIT_TEXT_LENGTH)
                                    binary_buffer = bytes(
                                        darwin3_device.FLIT_BINARY_LENGTH
                                    )
                                    darwin3_device.__gen_flit_parallel_east__(
                                        x,
                                        y,
                                        address,
                                        value,
                                        text_buffer,
                                        0,
                                        binary_buffer,
                                        0,
                                        config_list=config_list,
                                    )
                                    text_buffer = text_buffer * size
                                    binary_buffer = binary_buffer * size
                                else:
                                    buffer = write_f.read(length)
                                    index_byte = np.arange(0, length, bs)
                                    text_buffer = bytearray(
                                        size * darwin3_device.FLIT_TEXT_LENGTH
                                    )
                                    text_offset = np.arange(
                                        0,
                                        size * darwin3_device.FLIT_TEXT_LENGTH,
                                        darwin3_device.FLIT_TEXT_LENGTH,
                                    )
                                    binary_buffer = bytearray(
                                        size * darwin3_device.FLIT_BINARY_LENGTH
                                    )
                                    binary_offset = np.arange(
                                        0,
                                        size * darwin3_device.FLIT_BINARY_LENGTH,
                                        darwin3_device.FLIT_BINARY_LENGTH,
                                    )

                                    def convert(
                                        x,
                                        y,
                                        address,
                                        index_byte,
                                        text_offset,
                                        binary_offset,
                                    ):
                                        buffer_value = buffer[
                                            index_byte : index_byte + bs
                                        ] + b"\x00" * (8 - bs)
                                        value = struct.unpack("<Q", buffer_value)[0]
                                        darwin3_device.__gen_flit_parallel_east__(
                                            x,
                                            y,
                                            address,
                                            value,
                                            text_buffer,
                                            text_offset,
                                            binary_buffer,
                                            binary_offset,
                                            config_list=config_list,
                                        )

                                    np.frompyfunc(convert, 6, 0)(
                                        x,
                                        y,
                                        address,
                                        index_byte,
                                        text_offset,
                                        binary_offset,
                                    )
                                fin.write(text_buffer)
                                fbin.write(binary_buffer)
                elif (
                    item[1] == "write"
                    or item[1] == "write_ram"
                    or item[1] == "read_ack"
                    or item[1] == "write_risc"
                    or item[1] == "read_risc_ack"
                ) and os.path.exists(tc + item[5]):
                    if item[1] == "write_ram":
                        item[1] = "write"
                    tmp = item
                    x, y = int(item[2]), int(item[3])
                    addr = item[4]
                    while isinstance(addr, str):
                        addr = eval(addr)
                    with open(tc + item[5], "r") as write_f:
                        wlines = write_f.readlines()
                        wlength = len(wlines)
                        address = np.arange(addr, addr + wlength)
                        text_buffer = bytearray(
                            wlength * darwin3_device.FLIT_TEXT_LENGTH
                        )
                        text_offset = np.arange(
                            0,
                            wlength * darwin3_device.FLIT_TEXT_LENGTH,
                            darwin3_device.FLIT_TEXT_LENGTH,
                        )
                        binary_buffer = bytearray(
                            wlength * darwin3_device.FLIT_BINARY_LENGTH
                        )
                        binary_offset = np.arange(
                            0,
                            wlength * darwin3_device.FLIT_BINARY_LENGTH,
                            darwin3_device.FLIT_BINARY_LENGTH,
                        )

                        def convert(x, y, address, line, text_offset, binary_offset):
                            return darwin3_device.__gen_flit_parallel_east__(
                                x,
                                y,
                                address,
                                int(line, 16),
                                text_buffer,
                                text_offset,
                                binary_buffer,
                                binary_offset,
                                config_list=config_list,
                            )

                        np.frompyfunc(convert, 6, 0)(
                            x, y, address, wlines, text_offset, binary_offset
                        )
                        fin.write(text_buffer)
                        fbin.write(binary_buffer)
                else:
                    darwin3_device.__gen_flit_east__(
                        item,
                        fin,
                        fbin,
                        direct,
                        x_from=24,
                        y_from=-1,
                        config_list=config_list,
                    )
        load_f.close()

    def __flit_gen__(self, type="", input_file="", output_file="", update=True):
        config_list = {
            "last_vc": 1,
            "tick": 0,
            "start_tick": -1,
            "stop_tick": -1,
            "clear_tick": -1,
            "pkg_num": 0,
        }
        start_time = time.time_ns()
        print("===<0>=== generating %s.txt&%s.bin" %(output_file,output_file))
        file_path = ""
        if type == "deploy":
            file_path = self.deploy_path
        elif type == "input":
            file_path = self.input_path
        elif type == "debug":
            file_path = self.debug_path
        else:
            print("type not supported!")
            return
        if not os.path.exists(file_path + input_file):
            print("file not found!")
            return

        fin = open(file_path + output_file + ".txt", "wb")
        fin_bin = open(file_path + output_file + ".bin", "wb")
        config_list["tick"] = 0
        self.__gen_flit_by_fn__(
            file_path + input_file, fin, fin_bin, 0, file_path, config_list=config_list
        )
        fin.close()
        fin_bin.close()
        end_time = time.time_ns()
        print(
            "===<0>=== flit_gen elapsed : %.3f ms" % ((end_time - start_time) / 1000000)
        )

    def __flit_gen_east__(self, type="", input_file="", output_file="", update=True):
        config_list = {
            "last_vc": 1,
            "tick": 0,
            "start_tick": -1,
            "stop_tick": -1,
            "clear_tick": -1,
            "pkg_num": 0,
        }
        start_time = time.time_ns()
        print("===<0>=== generating %s.txt & %s.bin" % (output_file, output_file))
        file_path = ""
        if type == "deploy":
            file_path = self.deploy_path
        elif type == "input":
            file_path = self.input_path
        elif type == "debug":
            file_path = self.debug_path
        else:
            print("type not supported!")
            return
        if not os.path.exists(file_path + input_file):
            print("file not found!")
            return

        fin = open(file_path + output_file + ".txt", "wb")
        fin_bin = open(file_path + output_file + ".bin", "wb")
        config_list["tick"] = 0
        self.__gen_flit_by_fn_east__(
            file_path + input_file, fin, fin_bin, 0, file_path, config_list=config_list
        )
        fin.close()
        fin_bin.close()
        end_time = time.time_ns()
        print(
            "===<0>=== flit_gen elapsed : %.3f ms" % ((end_time - start_time) / 1000000)
        )

    def __gen_deploy_input_dwnc__(
        self,
        deploy_input_dwnc_file="deploy_input",
    ):
        """
        *-*-config.dwnc => deploy_input.dwnc
        Args:
            deploy_input_dwnc_file (str): 生成的 dwnc 文件的名称
        Returns:
            None
        """
        
        # 打开东西向传输的文件
        with open(self.deploy_path + deploy_input_dwnc_file + ".dwnc", "w+") as fwest, \
             open(self.deploy_path + deploy_input_dwnc_file + "_east.dwnc", "w+") as feast:
            
            # 清除神经元推理状态和权重和
            for neuron in self.config_neuron_list:
                if int(neuron[0]) <= 15:
                    fwest.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x04" "0x5"\n')
                else:
                    self.deploy_from_east = True
                    feast.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x04" "0x5"\n')
            
            # 将每个神经元的 config 文件整合到整体的 config 文件中
            search_paths = glob.glob(self.config_path + self.config_file_format)
            for search_path in search_paths:
                file = os.path.basename(search_path)
                x = re.findall(r"\d+", file)[0]
                if int(x) <= 15:
                    fwest.write("<< " + file + "\n")
                else:
                    self.deploy_from_east = True
                    feast.write("<< " + file + "\n")

            # 加入开启/停止tik控制对 => 代表配置结束
            fwest.write('0 cmd "0xc0000001"\n')
            fwest.write('0 cmd "0xc0000000"\n')
            
            feast.write('0 cmd "0xc0000001"\n')
            feast.write('0 cmd "0xc0000000"\n')

            # 设置tick
            fwest.write('0 cmd "0xe{:0>7x}"\n'.format(self.step_size))
            feast.write('0 cmd "0xe{:0>7x}"\n'.format(self.step_size))

            # 使能神经元，清除神经元权重和
            for neuron in self.config_neuron_list:
                if int(neuron[0]) <= 15:
                    fwest.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x15" "0x1"\n')
                    fwest.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x04" "0x1"\n')
                else:
                    self.deploy_from_east = True
                    feast.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x15" "0x1"\n')
                    feast.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x04" "0x1"\n')
        return

    def __gen_deploy_flitin__(
        self,
        deploy_input_dwnc_file="deploy_input",
        deploy_flitin_file="deploy_flitin",
    ):
        """
        deploy_input.dwnc => deploy_flitin.txt && deploy_flitin.bin
        Args:
            deploy_input_dwnc_file (str): 部署使用的 dwnc 文件名称
            deploy_flitin_file (str): 生成的部署使用的 flit 文件名称
        Returns:
            None
        """
        self.__flit_gen__(
            type="deploy",
            input_file=deploy_input_dwnc_file+".dwnc",
            output_file=deploy_flitin_file,
        )
        if self.deploy_from_east:
            self.__flit_gen_east__(
                type="deploy",
                input_file=deploy_input_dwnc_file+"_east.dwnc",
                output_file=deploy_flitin_file+"_east"
            )
        return

    def __gen_spike_file__(
        self, spike_neurons: list, spike_file="spikes.dwnc"
    ):
        """
        input_neuron.json && spike_neurons (list) => spikes.dwnc
        根据输入的每个时间步的脉冲信息, 结合每个神经元的配置 json 文件, 生成 dwnc 文件配置
        Args:
            spike_neurons (list): 输入的神经元脉冲序列
            spike_file (str): 生成的脉冲输入 dwnc 文件配置
        Returns:
            None
        """
        with open(self.config_path + spike_file, "w+") as f:
            for i in range(0, len(spike_neurons)):
                time_step = i + 1
                cur_spike_neuron_list = spike_neurons[i]
                for spike_neuron in cur_spike_neuron_list:
                    neuron_info = self.input_neuron[str(spike_neuron)]
                    if len(neuron_info) > 0:
                        neuron_type = neuron_info[0]
                        targets_list = neuron_info[-1]
                        if neuron_type == 0:
                            neu_idx = hex(neuron_info[1])   # 返回'0x'开头的字符串
                        elif neuron_type == 1:
                            neu_idx = "0x0"
                        for target in targets_list:
                            x = target[0]
                            y = target[1]
                            derd_id = hex(target[2])
                            cur_line = str(time_step) + " spike " + str(x) + " " + str(y) + " \"" + derd_id + "\""+" \""+ neu_idx +"\"\n"
                            f.write(cur_line)
        return

    def __gen_run_input_dwnc__(
        self,
        spike_neurons: list,
        spike_file="spikes.dwnc",
        run_input_dwnc_file="run_input.dwnc",
    ):
        """
        input_neuron.json && length of spike_neurons (list) => spikes.dwnc
        跟据spikes.dwnc以及config文件中提到的神经元，生成对应的run_input.dwnc文件
        Args:
            spike_neurons (list): 输入的神经元脉冲序列 (仅用到其length, 可以优化)
            spike_file (str): 脉冲输入 dwnc 文件配置
            run_input_dwnc_file (str): 生成的运行 dwnc 文件
        Returns:
            None
        """
        steps = len(spike_neurons)
        with open(self.input_path + run_input_dwnc_file, "w+") as f:
            f.write('0 cmd "0xc0000001"\n')
            f.write('<< ' + spike_file + '\n')
            f.write(str(steps) + ' cmd "0xc0000000"\n')
        with open(self.input_path + 'sync.dwnc', 'w+') as f:
            f.write('0 cmd "0xc0000001"\n')
            f.write(str(steps) + ' cmd "0xc0000000"\n')
        return

    def __gen_run_flitin__(
        self, run_input_dwnc_file="run_input.dwnc", run_flitin_file="run_flitin"
    ):
        """
        run_input.dwnc => run_flitin.txt && run_flitin.bin
        Args:
            run_input_dwnc_file (str): 输入的运行 dwnc 文件
            run_flitin_file (str): 生成的运行 flit 文件
        Returns:
            None
        """
        self.__flit_gen__(
            type="input",
            input_file=run_input_dwnc_file,
            output_file=run_flitin_file,
        )
        self.__flit_gen__(
            type="input",
            input_file="sync.dwnc",
            output_file="sync",
        )
        return

    def __transmit_flit__(self, port, data_type, west_fbin, east_fbin, north_fbin, south_fbin, recv=False, recv_run_flit_file="recv_run_flit", debug=False):
        """
        发包到darwin3, recv=True时接收darwin3返回来的包
        Args:
            port (list(int)): TCP 连接端口列表
            data_type (int): 发送包的格式
            freq (int): 设置的时钟频率 (仅当 data_type==SET_FREQUENCY 时有效)
            fbin (str): 发送的包内容 (仅当 data_type==NORMAL_FLIT 时有效)
            recv (bool): 是否接受 Darwin3 的返回包
            recv_run_flit_file (str): 保存返回包的名称
            debug (bool): 调试标记
        Returns:
            None
        """
        trans = Transmitter()
        ip_address = (self.ip, port)
        trans.connect_lwip(ip_address)
        print("===<1>=== tcp connect succeed")
        start_time = time.time_ns()
        trans.send_flit_data(
            trans.pack_flit_data(data_type, west_fbin, east_fbin, north_fbin, south_fbin)
        )
        # trans.send_flit_bin(fbin, data_type)
        print("===<2>=== send succeed")
        end_time = time.time_ns()
        print('===<3>=== tcp sent elapsed : %.3f ms' % ((end_time - start_time)/1000000))
        if recv:
            self.__recv_flit__(trans,recv_run_flit_file,debug)
        trans.close()
        return

    def __recv_flit__(self, trans, recv_run_flit_file="recv_run_flit", debug=False):
        """
        接收从darwin3来的包
                Args:
            trans (Transmitter): Transmitter
            recv_run_flit_file (str): 保存返回包的名称
            debug (bool): 调试标记
        Returns:
            None
        """
        _recv_buffer=trans.socket_inst.recv(10240)
        if len(_recv_buffer)<16:
            return
        _west_len,_east_len,_north_len,_south_len=struct.unpack('IIII',_recv_buffer[0:16])
        _index=16
        _west_data=_recv_buffer[_index:_index+_west_len]
        _index+=_west_len
        _east_data= _recv_buffer[_index:_index+_east_len]
        _index+=_east_len
        _north_data=_recv_buffer[_index:_index+_north_len]
        _index+=_north_len
        _south_data=_recv_buffer[_index:_index+_south_len]
        self._write_received_flit_to_file(_west_data,'west',recv_run_flit_file,debug)
        self._write_received_flit_to_file(_east_data,'east',recv_run_flit_file,debug)
        self._write_received_flit_to_file(_north_data,'north',recv_run_flit_file,debug)
        self._write_received_flit_to_file(_south_data,'south',recv_run_flit_file,debug)


    def _write_received_flit_to_file(self,flit_data,direction:str,recv_run_flit_file,debug):
        if debug:
            file_path = self.debug_path
        else:
            file_path = self.output_path
        fout = open(os.path.join(file_path,f'{recv_run_flit_file}_{direction}.txt'), "wb")
        foutbin = open(os.path.join(file_path,f'{recv_run_flit_file}_{direction}.bin'), "wb")
        hl = b""
        index = 0
        tot = 0
        
        foutbin.write(flit_data)
        for i in range(len(flit_data)):
            b = b"%02x" % flit_data[i]
            hl = b + hl
            index = index + 1
            if (index == 4):
                fout.write (hl + b"\n")
                # print(hl)
                hl = b""
                index = 0
                tot = tot + 1
        fout.close()
        foutbin.close()

    def __run_parser__(self, recv_run_flit_file='recv_run_flit', result=[[],], debug=False, pop_name='',direction='west'):
        """
        解析 Darwin3 返回的包
        Args:
            recv_run_flit_file (str): 返回包的名称
            result (list(list)): 解析结果
            debug (bool): 调试标记
            pop_name (str): 输出的文件名
        Returns:
            None
        """
        print('===<4>=== parser recv flit : %s.txt' % recv_run_flit_file)
        if debug:
            file_path = self.debug_path
        else:
            file_path = self.output_path
        start_time = time.time_ns()
        with open(file_path+recv_run_flit_file+'.txt', 'r') as flit_f :
            lines = flit_f.readlines()
            index = 0
            is_write = 0
            is_spike = 0
            is_reward = 0
            is_flow = 0
            is_dedr = 0
            t = 1
            if debug:
                fanalyse = open(file_path+recv_run_flit_file+'_'+pop_name+'_analyse.txt','w')
            for items in lines:
                if (eval("0x"+items[0])>>2)&0x3 == 3:  #flit_type=3 
                    cmd = eval("0x"+items[0:2])&0x3f    #[31:26]
                    if cmd == 0b011000:     # d8000000
                        arg = eval("0x"+items[2:8])
                        t+=arg+1
                elif (eval("0x"+items[0])>>2)&0x3 == 2 :    #flit_type=2 包头
                    index = 0
                    is_write = (eval("0x"+items[1:3])>>2) & 0x7 == 1    #[24:22]
                    is_spike = (eval("0x"+items[1:3])>>2) & 0x7 in (0,4,5,6)
                    is_reward= (eval("0x"+items[1:3])>>2) & 0x7 in (5,6)
                    is_flow  = (eval("0x"+items[1:3])>>2) & 0x7 == 7
                    dst_x = (eval("0x"+items[3:5])>>2) & 0xf    #[17:14]
                    if (eval("0x"+items[3:5])>>6) & 0x1 == 1:     #[18]
                        dst_x = -dst_x
                    if direction == 'west':
                        y = (eval("0x"+items[0:2])>>1)&0x1f     #[29:25]
                        x = ((eval("0x"+items[5:7])>>1)&0xf) + dst_x - 1    #[8:5]
                    elif direction == 'east':
                        y = (eval("0x"+items[0:2])>>1)&0x1f     #[29:25]
                        x = 24 - ((eval("0x"+items[5:7])>>1)&0xf) + dst_x    #[8:5]
                elif (is_write == 1) :
                    if (eval("0x"+items[0])>>2)&0x3 == 1 :
                        value = (value<<24) + ((eval("0x"+items[0:8])>>3)&0x7ffffff)
                        addr_eff = addr & 0x1ffff
                        addr_relay = (addr >> 18) & 0x3f
                        if (is_dedr):
                            print ("[dedr] tik=%d, x=%d, y=%d, relay_link=0x%02x, addr=0x%05x, value=0x%012x " % (t,x,y,addr_relay,addr_eff,value))
                        else:
                            if addr_eff >= 0x8000:
                                c = "axon"
                            elif addr_eff >= 0x4000:
                                c = "wgtsum"
                            elif addr_eff >= 0x2000:
                                c = "vt"
                            elif addr_eff >= 0x1000:
                                c = "inst"
                            elif addr_eff >= 0x800:
                                c = "reg"
                            else:
                                c = "conf"
                            v = value & 0xffff
                            if v >= 0x8000 and addr_eff >= 0x800 and addr_eff < 0x8000:
                                v = v - 0x10000
                            print ("[%s] tik=%d, x=%d, y=%d, relay_link=0x%02x, addr=0x%05x, value=0x%08x (%d)" % (c,t,x,y,addr_relay,addr_eff,value,v))
                            addr_eff = f"{addr_eff:05x}"
                            value = f"{value:08x}"
                            if debug:
                                fanalyse.write('\"0x'+addr_eff+'\" \"0x'+value+'\"'+' '+str(v)+'\n')
                        is_write = 0
                    elif (eval("0x"+items[0])>>2)&0x3 == 0 :
                        if (index == 0):
                            addr = (eval("0x"+items[0:8])>>3)&0x7ffffff
                            index = index + 1
                            if ((addr & 0x1ffff) >= 0x10000):
                                is_dedr = 1
                        else :
                            value = (eval("0x"+items[0:8])>>3)&0x7ffffff
                elif (is_spike == 1) :
                    if (eval("0x"+items[0])>>2)&0x3 == 1 :      #flit_type=1 包尾
                        neu_idx = (eval("0x"+items[4:8]) >> 3) & 0xfff  #[29:15]
                        dedr_id = (eval("0x"+items[0:5]) >> 3) & 0x7fff  #[14:3]

                        format = "output_neuron*.json"
                        pattern = re.compile(r'output_neuron_(.*?)\.json')
                        search_paths = glob.glob(self.neuron_path+format)
                        index = str(x)+", "+str(y)+", "+str(dedr_id)
                        for search_path in search_paths:
                            file = os.path.basename(search_path)
                            output_name = pattern.match(file).group(1)
                            with open(self.neuron_path + file, "r") as f:
                                output_neuron_info = json.load(f)
                                if output_neuron_info.get(index) is not None:
                                    result[t-1].append((output_name,output_neuron_info[index]))
                        if is_reward:
                            print ("[rwd] tik=%d, x=%d, y=%d, dedr_id=0x%04x, wgt=0x%02x" % (t,x,y,dedr_id,neu_idx))
                        else:   
                            print ("[spk] tik=%d, x=%d, y=%d, dedr_id=0x%04x, neu_idx=0x%03x" % (t,x,y,dedr_id,neu_idx))
                        is_spike = 0
                        is_reward = 0
                elif (is_flow == 1) :
                    if (eval("0x"+items[0])>>2)&0x3 == 1 :
                        addr_relay = (addr >> 18) & 0x3f
                        data = data + ((eval("0x"+items[0:8])&0x3fffffff)<<24)
                        data = (data << 17) + (addr & 0x1ffff)
                        print ("[flow] tik=%d, x=%d, y=%d, relay_link=0x%02x, data=0x%018x" % (t,x,y,addr_relay,data))
                        is_flow = 0
                    elif (eval("0x"+items[0])>>2)&0x3 == 0 :
                        if (index == 0):
                            addr = (eval("0x"+items[9:16])>>3)&0x7ffffff
                            index = index + 1
                        else :
                            data = (eval("0x"+items[0:8])>>3)&0x7ffffff
            if debug:
                fanalyse.close()
        flit_f.close()
        end_time = time.time_ns()
        print('===<4>=== parser recv flit elapsed : %.3f ms' % ((end_time - start_time)/1000000))
        return


    def reset(self):
        """
        复位硬件接口相关逻辑和硬件系统(darwin3 芯片, DMA 等)
        Args: 
            None
        Returns:
            None
        """
        requests.get(self.control_url+'/chip_reset')
        print("Please check the information on the Darwin3 development board ")
        print("to determine if the configuration was successful.")
        return

    def darwin3_init(self, freq=333):
        """
        按照指定频率配置 darwin3 芯片。
        Args:
            freq (int): 频率 (默认 333MHz, 仅支持 20MHz 和 333MHz)
        Returns:
            None
        """
        requests.get(self.control_url+f'/set_frequency?freq={str(freq)}')
        print("Please check the information on the Darwin3 development board ")
        print("to determine if the configuration was successful.")
        return 0

    def deploy_config(self):
        """
        在部署芯片上部署并使能相关核心, 同时清除神经元的相关状态
        Args:
            None
        Returns:
            None
        """
        self.__gen_deploy_input_dwnc__()
        self.__gen_deploy_flitin__()
        self.__transmit_flit__(port=self.port[0], data_type=self.DEPLOY_FLIT, west_fbin=self.deploy_path+"deploy_flitin.bin",
                                east_fbin=self.deploy_path+"deploy_flitin_east.bin",north_fbin=b'',south_fbin=b'')
        return

    def run_darwin3_withoutfile(self, spike_neurons: list, outdir="west"):
        """
        接收应用给的 spike_neurons 作为输入，运行 len(spike_neurons) 个时间步
        此函数不产生中间文件
        Args:
            spike_neurons (list): sequence, 本次应用输入给硬件的脉冲数据, 
                                  序列长度与时间步数量一致，没有脉冲的时间步给空值
            outdir (str): 脉冲输出的方向, 支持 east 和 west 方向的输出, 默认 west
        Returns:
            result (list): 本次运行结束时硬件返回给应用的脉冲
        """
        result = [[] for _ in range(len(spike_neurons))]
        files = [self.output_path+'recv_run_flit.bin', self.output_path+'recv_run_flit.txt', self.config_path+'spikes.dwnc', self.input_path+'run_flitin.bin', self.input_path+'run_flitin.txt', self.input_path+'run_input.dwnc']
        for file in files:
            try:
                os.remove(file)
            except:
                pass
        with Patcher() as patcher:
            # patcher.fs.add_real_directory(self.app_path)
            # patcher.fs.add_real_directory(self.neuron_path)
            # patcher.fs.add_real_directory(self.input_path)
            # patcher.fs.add_real_directory(self.output_path)
            patcher.fs.makedir(self.app_path)
            patcher.fs.makedir(self.input_path)
            patcher.fs.makedir(self.output_path)
            patcher.fs.makedir(self.config_path)
            # patcher.fs.makedir(self.debug_path)
            # patcher.fs.add_real_paths(self.config_path)
            # patcher.fs.add_real_file(self.config_path + "spikes.dwnc")
            self.__gen_spike_file__(spike_neurons)
            self.__gen_run_input_dwnc__(spike_neurons)
            self.__gen_run_flitin__()
            self.__transmit_flit__(port=self.port[0], data_type=self.RUN_FLIT, west_fbin=self.input_path+"run_flitin.bin",east_fbin=self.input_path+"sync.bin",
                                north_fbin=b'',south_fbin=b'',recv=True, recv_run_flit_file="recv_run_flit")
            self.__run_parser__(result=result, direction = outdir, recv_run_flit_file='recv_run_flit_'+outdir)
            with open(self.output_path + "recv_run_flit.txt",'r') as f:
                recv_run_flit_txt = f.read()
            with open(self.output_path + "recv_run_flit.bin",'rb') as f:
                recv_run_flit_bin = f.read()
        with open(self.output_path + "recv_run_flit.txt",'w') as f:
            f.write(recv_run_flit_txt)
        with open(self.output_path + "recv_run_flit.bin",'wb') as f:
            f.write(recv_run_flit_bin)

        return result
    
    def run_darwin3_withfile(self, spike_neurons: list, outdir="west"):
        """
        接收应用给的 spike_neurons 作为输入，运行 len(spike_neurons) 个时间步
        此函数会产生中间文件进行过渡
        Args:
            spike_neurons (list): sequence, 本次应用输入给硬件的脉冲数据, 
                                  序列长度与时间步数量一致，没有脉冲的时间步给空值
            outdir (str): 脉冲输出的方向, 支持 east 和 west 方向的输出, 默认 west
        Returns:
            result (list): 本次运行结束时硬件返回给应用的脉冲
        """
        result = [[] for _ in range(len(spike_neurons))]
        self.__gen_spike_file__(spike_neurons)
        self.__gen_run_input_dwnc__(spike_neurons)
        self.__gen_run_flitin__()
        self.__transmit_flit__(port=self.port[0], data_type=self.RUN_FLIT, west_fbin=self.input_path+"run_flitin.bin",east_fbin=self.input_path+"sync.bin",
                               north_fbin=b'',south_fbin=b'',recv=True, recv_run_flit_file="recv_run_flit")
        self.__run_parser__(result=result, direction = outdir, recv_run_flit_file='recv_run_flit_'+outdir)
        return result

    def dump_config(self, x: int, y: int, config="all", cfg_o=""):
        """
        !!! Not yet implemented
        根据核心坐标和类型查询并下载芯片上的配置信息到本地，用于与编译结果比对以便确认部署阶段的正确运行。
        (NoC 不保证传输顺序，如果存在不一致需要进一步确认)
        Args:
            x (int): 核心的 x 坐标
            y (int): 核心的 y 坐标
            config (str): 包括"reg", "axon", "dendrite", "instruction" 或者 “all”
            cfg_o: file: 接收后产生的 x-y-xx.txt 文件名称
        Returns:
            None
        """
        print("!!! Not yet implemented")
        return

    def get_neuron_state(self, pop_name: str, state: list, offset=0):
        """
        从硬件获取神经元的状态，状态空间包括膜电位、权重和、推理参数、推理状态等，主要用于调试。
        Args:
            pop_name (str): 编译阶段的 Population Name
            state (list): sequence[array]，包含一组或多组[神经元序号, [状态空间列表]]
                   膜电位(vt) => "read", x, y, neuron_index+0x02000  按升序
                   权重和(wgtsum) => "read", x, y, neuron_index+0x04000  按升序
                   推理参数(inference_parameter) => "read", x, y, neuron_index+0x1E000(按升序)  0x1EFFF-neuron_index(按降序)
                   推理状态(inference_status) => "read", x, y, neuron_index+0x1F000(按升序)  0x1FFFF-neuron_index(按降序)
            offset (int): 用户指定的地址偏移
        Returns:
            None
        """
        debug_from_east = False
        debug_from_west = False
        with open(self.neuron_path + pop_name +".json", "r") as f:
            hidden_info = json.load(f)

        with open(self.debug_path+'get_neuron_state_input.dwnc', 'w') as fwest,\
            open(self.debug_path+'get_neuron_state_input_east.dwnc', 'w') as feast:
            fwest.write('0 cmd "0xc0000001"\n')
            feast.write('0 cmd "0xc0000001"\n')
            for neu_idx, state_space in state:
                targetlist = hidden_info[str(neu_idx)]
                x = targetlist[0]
                y = targetlist[1]
                neu_offset = targetlist[2]
                for name in state_space:
                    if name == 'vt':
                        item = f"0 read {x} {y} \""+ hex(neu_offset+0x02000)+"\" 1\n"
                    elif name == 'wgtsum0':
                        item = f"0 read {x} {y} \""+ hex(neu_offset+0x04000)+"\" 1\n"
                    elif name == 'wgtsum1':
                        item = f"0 read {x} {y} \""+ hex(neu_offset+0x05000)+"\" 1\n"
                    elif name == 'inference_parameter':
                        item = f"0 read {x} {y} \""+ hex(0x1EFFF-neu_offset)+"\" 1\n"
                    elif name == 'inference_status':
                        item = f"0 read {x} {y} \""+ hex(0x1FFFF-neu_offset)+"\" 1\n"
                    elif name == 'npu_reg':
                        item = f"0 read {x} {y} \""+ hex(0x00800 + offset)+"\" 1\n"
                    elif name == 'config_reg':
                        item = f"0 read {x} {y} \""+ hex(0x00000 + offset)+"\" 1\n"
                    elif name == 'dedr':
                        item = f"0 read {x} {y} \""+ hex(0x10000 + offset)+"\" 1\n"
                    else:
                        print('error! not allowed %s', name)
                    # else:
                    #     item = f"0 read {x} {y} \""+ hex(0x01000+neu_offset)+"\" 1\n"
                    if x>15:
                        feast.write(item)
                        debug_from_east = True
                    else:
                        fwest.write(item)
                        debug_from_west = True
            fwest.write('0 cmd "0xc0000000"\n')
            feast.write('0 cmd "0xc0000000"\n')

        self.__flit_gen_east__(type="debug", input_file='get_neuron_state_input_east.dwnc', output_file='get_neuron_state_flitin_east')
        self.__flit_gen__(type="debug", input_file='get_neuron_state_input.dwnc', output_file='get_neuron_state_flitin')
        self.__transmit_flit__(port=self.port[0], data_type=self.STATE_FLIT, west_fbin=self.debug_path+"get_neuron_state_flitin.bin", east_fbin=self.debug_path+'get_neuron_state_flitin_east.bin', north_fbin=b'', south_fbin=b'', recv=True, recv_run_flit_file="get_neuron_state_recv", debug=True)
        if debug_from_east == True:
            self.__run_parser__(recv_run_flit_file='get_neuron_state_recv_east', debug=True, pop_name=pop_name, direction='east')
        if debug_from_west == True:
            self.__run_parser__(recv_run_flit_file='get_neuron_state_recv_west', debug=True, pop_name=pop_name, direction='west')

        debug_from_west == False
        debug_from_east == False

        return

    def set_neuron_state(self, pop_name: str, state: list, value: list):
        """
        !!! Not yet implemented
        设置硬件神经元的状态，包括膜电位、权重和、推理参数、推理状态等，主要用于调试。
        Args:
            pop_name (str): 编译阶段的 Population Name
            state (list(str, list)): 包含一组或多组[神经元序号, [状态空间列表]]
            value (list): 状态空间对应的所需要设置的内容
        Returns:
            None
        """
        print("!!! Not yet implemented")
        return

    def enable_neurons(self, dwnc_file="enable"):
        """
        将 config_files 中所有需要使用的神经元使能
        Args:
            dwnc_file (str): 生成的配置文件名称
        Returns:
            None
        """
        
        # 生成 dwnc 文件
        with open(self.deploy_path + dwnc_file + ".dwnc", "w+") as fwest, \
        open(self.deploy_path + dwnc_file + "_east.dwnc", "w+") as feast:
            for neuron in self.config_neuron_list:
                if int(neuron[0]) <= 15:
                    fwest.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x15" "0x1"\n')
                else:
                    self.deploy_from_east = True
                    feast.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x15" "0x1"\n')
            fwest.write('0 cmd "0xc0000001"\n')
            feast.write('0 cmd "0xc0000001"\n')
            fwest.write('0 cmd "0xc0000000"\n')
            feast.write('0 cmd "0xc0000000"\n')
        
        # 生成 flit 文件
        self.__flit_gen__(
            type="deploy",
            input_file=dwnc_file+".dwnc",
            output_file=dwnc_file+"_flitin",
        )
        self.__flit_gen_east__(
            type="deploy",
            input_file=dwnc_file+"_east.dwnc",
            output_file=dwnc_file+"_flitin_east",
        )
        
        # 发送 flit 包到 Darwin3 板卡
        self.__transmit_flit__(port=self.port[0], data_type=self.DEPLOY_FLIT, west_fbin=self.deploy_path+dwnc_file+"_flitin.bin",
                        east_fbin=self.deploy_path+dwnc_file+"_flitin_east.bin",north_fbin=b'',south_fbin=b'')
        return
    
    def disable_neurons(self, dwnc_file="disable"):
        """
        将 config_files 中所有需要使用的神经元取消使能
        Args:
            dwnc_file (str): 生成的配置文件名称
        Returns:
            None
        """
        
        # 生成 dwnc 文件
        with open(self.deploy_path + dwnc_file + ".dwnc", "w+") as fwest, \
        open(self.deploy_path + dwnc_file + "_east.dwnc", "w+") as feast:
            for neuron in self.config_neuron_list:
                if int(neuron[0]) <= 15:
                    fwest.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x15" "0x1"\n')
                else:
                    self.deploy_from_east = True
                    feast.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x15" "0x1"\n')
            fwest.write('0 cmd "0xc0000001"\n')
            feast.write('0 cmd "0xc0000001"\n')
            fwest.write('0 cmd "0xc0000000"\n')
            feast.write('0 cmd "0xc0000000"\n')
            
        # 生成 flit 文件
        self.__flit_gen__(
            type="deploy",
            input_file=dwnc_file+".dwnc",
            output_file=dwnc_file+"_flitin",
        )
        self.__flit_gen_east__(
            type="deploy",
            input_file=dwnc_file+"_east.dwnc",
            output_file=dwnc_file+"_flitin_east",
        )
            
        # 发送 flit 包到 Darwin3 板卡
        self.__transmit_flit__(port=self.port[0], data_type=self.DEPLOY_FLIT, west_fbin=self.deploy_path+dwnc_file+"_flitin.bin",
                        east_fbin=self.deploy_path+dwnc_file+"_flitin_east.bin",north_fbin=b'',south_fbin=b'')
        return
        
    def clear_neurons_states(self, ISC=False, LSC=False, clear=False, dwnc_file="clear_states"):
        """
        清理 darwin3 芯片内部神经拟态核心的状态量
        Args:
            ISC   (bool): inference status clear,
                          推理状态中电流清零, 阈值和振荡电位复位, 1 有效
                          相关配置寄存器: dedr_vth_keep, dedr_vth_gset, 
                          global_vth, dedr_res_keep, global_res
            LSC   (bool): learn status clear, 学习状态清零, 1 有效
            clear (bool): 权重和清零, 膜电位复位, 1 有效
                          相关配置寄存器:vt_rest
                          
            dwnc_file (str): 生成的配置文件名称
        Returns:
            None
        """
        
        # 根据需要重置的内容生成指令
        clear_type = hex(int(''.join(str(int(b)) for b in [ISC, LSC, clear]), 2))
        
        # 生成 dwnc 文件
        with open(self.deploy_path + dwnc_file + ".dwnc", "w+") as fwest, \
        open(self.deploy_path + dwnc_file + "_east.dwnc", "w+") as feast:
            for neuron in self.config_neuron_list:
                if int(neuron[0]) <= 15:
                    fwest.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x04" "' + clear_type + '"\n')
                else:
                    self.deploy_from_east = True
                    feast.write("0 write " + neuron[0] + " " + neuron[1] + ' "0x04" "' + clear_type + '"\n')
            fwest.write('0 cmd "0xc0000001"\n')
            feast.write('0 cmd "0xc0000001"\n')
            fwest.write('0 cmd "0xc0000000"\n')
            feast.write('0 cmd "0xc0000000"\n')
            
        # 生成 flit 文件
        self.__flit_gen__(
            type="deploy",
            input_file=dwnc_file+".dwnc",
            output_file=dwnc_file+"_flitin",
        )
        self.__flit_gen_east__(
            type="deploy",
            input_file=dwnc_file+"_east.dwnc",
            output_file=dwnc_file+"_flitin_east",
        )
            
        # 发送 flit 包到 Darwin3 板卡
        self.__transmit_flit__(port=self.port[0], data_type=self.DEPLOY_FLIT, west_fbin=self.deploy_path+dwnc_file+"_flitin.bin",
                        east_fbin=self.deploy_path+dwnc_file+"_flitin_east.bin",north_fbin=b'',south_fbin=b'')
        return