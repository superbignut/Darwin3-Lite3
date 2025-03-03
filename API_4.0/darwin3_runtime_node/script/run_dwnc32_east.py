import os
import pickle
import sys
sys.path.append("../../compiler")
import numpy as np
import random
import socket
import struct
import time
import argparse
import itertools

from darwinc.sir.physical_graph import PhysicalGraph
from darwinc.sir.graph_mapper import GraphMapper

from darwinc.sir.machine import Machine
from darwinc.sir.chip_node import ChipNode
from darwinc.frontend.import_regular_net_graph import ImportDAGNetGraph
from darwinc.algorithms.node_allocator_algorithm import DAGSimpleAllocator
from darwinc.backend.code_gen import CodeGen
from darwinc.constraints.node_constraints import SmallNodeConstraints

last_vc = 1
tick    = 0
start_tick = -1
stop_tick = -1
clear_tick = -1
pkg_num = 0

class Transmitter(object):
    def __init__(self):
        self.socket_inst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_inst.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect_lwip(self, ip_address):
        self.socket_inst.connect(ip_address)

    def close(self):
        self.socket_inst.close()
        
    def send_flit_bin(self, flit_bin_file):
        '''
        发送flit
        '''
        with open(flit_bin_file, 'rb') as file:
            flit_bin = file.read()
        length = len(flit_bin) >> 2
        if length > 2**26:
            print("===<2>=== %s is larger than 0.25GB" % flit_bin_file)
            print("===<2>=== send flit length failed")
            return 0
        send_bytes = bytearray()
        send_bytes += struct.pack('I', length)
        send_bytes += struct.pack('I', 0x8000)
        send_bytes += flit_bin
        #self.socket_inst.sendall(struct.pack('I', length))
        #ack = self.socket_inst.recv(1024)
        #if (ack == b'done'):
        #    print("===<2>=== send flit length succeed")
        #self.socket_inst.sendall(flit_bin)
        self.socket_inst.sendall(send_bytes)
        return 1

    def send_flit(self, flit_file, directions=0):
        '''
        发送flit
        '''
        with open(flit_file, 'r') as file:
                flit_list = file.readlines()
        length = len(flit_list)
        if length > 2**26:
            print("===<2>=== %s is larger than 0.25GB" % flit_file)
            print("===<2>=== send flit length failed")
            return 0
        print("===<2>=== send flit length succeed")
        #self.socket_inst.sendall(struct.pack('I', length))
        #ack = self.socket_inst.recv(1024)
        #if (ack == b'done'):
        
        j = 0
        while(j < length):
            send_bytes = bytearray()
            send_bytes += struct.pack('I', length)
            send_bytes += struct.pack('I', 0x8000)
            for i in range(j,min(j + 16777216 * 4,length)):
                send_bytes += struct.pack('I', int(flit_list[i % length].strip(),16))
            self.socket_inst.sendall(send_bytes)
            j = j + 16777216 * 4
            if (j <= length):
                reply = self.socket_inst.recv(1024)
                print("%s" % reply)
        return 1    

def onehot2bin(port):
    if (port == 1):
        return 0
    if (port == 2):
        return 1
    if (port == 4):
        return 2
    if (port == 8):
        return 3
    if (port == 16):
        return 4
    return 0

def fpga_machine():
    # ignore the relay link
    
    X = [23, 19, 15, 11]
    Y = [5, 7, 13, 15, 21, 23]
    available_coord = set(itertools.product(X, Y))

    fpga_machine = Machine(24, 24)
    for coord in fpga_machine.nodes.keys():
        if not coord in available_coord:
            fpga_machine.nodes[coord].is_usable = False
        else:
            if (coord[0] + coord[1]) % 8 == 2 or (coord[0] + coord[1]) % 8 == 4:
                fpga_machine.nodes[coord].constraints = SmallNodeConstraints()

    # add virtual input node
    fpga_machine.set_node((24, 13), ChipNode((24, 13)))
    fpga_machine.node((24, 13)).is_input = True
    fpga_machine.input_node_coord = (24, 13)
 
    # add virtual output node
    fpga_machine.set_node((24, 15), ChipNode((24, 15)))
    fpga_machine.node((24, 15)).is_output = True
    fpga_machine.output_node_coord = (24, 15)
    return fpga_machine

def case_from_file(fn="connections.data", out="darwin2_gen", rsm=2, gt=False, vth=60):

    # load connections from file
    with open(fn,'rb') as f:
        connections = pickle.load(f)

    # import the given connections into net_graph
    net_graph = ImportDAGNetGraph.make_graph(connections, 'simple_net')
    phy_graph = PhysicalGraph('simple_net_phy')

    # define graph mapper to link the net_graph and physical graph
    graph_mapper = GraphMapper(net_graph, phy_graph)
    machine = fpga_machine()

    # define algorithms
    algorithm = DAGSimpleAllocator()
    algorithm.allocate(graph_mapper, machine)
    
    # define code gen, allocate memory space for each node in the machine
    code_gen = CodeGen()
    code_gen.fill_machine_memory(machine, graph_mapper)
    
    # cur_path = os.path.dirname(os.path.abspath(__file__))
    # fpath = os.path.join(cur_path, out)
    fpath = out
    if not os.path.exists(fpath):
        os.mkdir(fpath)

    if os.path.exists('config.dwnc'):
        os.remove(os.path.join(fpath, 'config.dwnc'))

    if os.path.exists('clear.dwnc'):
        os.remove(os.path.join(fpath, 'clear.dwnc'))
        
    if os.path.exists('enable.dwnc'):
        os.remove(os.path.join(fpath, 'enable.dwnc'))
        
    if os.path.exists('disable.dwnc'):
        os.remove(os.path.join(fpath, 'disable.dwnc'))
        
    code_gen.serialize(machine, fpath, rsm, gt, vth)

    # save connections to json
    if fpath != ".":
        with open(os.path.join(fpath, 'connections.data'), 'wb') as fid:
            pickle.dump(connections, fid)

    log_fname = os.path.join(fpath, 'loc.pkl')
    loc = {}
    with open(log_fname, 'wb') as fid:
        for pop_label, pop in graph_mapper.phy_graph.pops.items():
            # print(pop_label)
            #fid.write(str(sorted(pop.neurons)) + ' ' + str(pop.node) + '\n')
            x,y = pop.node
            if x>=0 and x<24 and y>=0:
                loc[pop.node] = pop.neurons
        pickle.dump(loc, fid)

def gen_config_dwnc(fconn, fout='.', rsm=2, gt=False, vth=60):
    if fconn is not None:
        start_time = time.time_ns()
        print("===<0>=== generating config.dwnc")
        case_from_file(fconn, fout, rsm, gt, vth)
        end_time = time.time_ns()
        print('===<0>=== gen_config_dwnc elapsed : %.3f ms' % ((end_time - start_time)/1000000))

def find_dedr_id(neu_id, axon_lines, x_from=-1, y_from=13):
    axon_id = (eval("0x"+axon_lines[neu_id])) >> 12
    neu_idx = (eval("0x"+axon_lines[neu_id])) & 0xfff
    ret = []
    while True:
        axon = eval("0x"+axon_lines[axon_id])
        dedr_id = axon & 0x7FFF
        y = (axon >> 15) & 0xF
        y_sign = (axon >> 19) & 0x1
        if y_sign == 1:
            y = -y
        x = (axon >> 20) & 0xF
        x_sign = (axon >> 24) & 0x1
        if x_sign == 1:
            x = -x
        lnf = axon >> 25
        ret.append((x+x_from,y+y_from,dedr_id,neu_idx))
        if (lnf == 1):
            break
        axon_id += 1
    return ret

def gen_spike(spks, axon_lines, spk_file='spk.dwnc'):
    with open(spk_file,'w') as fspk:
        __stdout__ = sys.stdout
        sys.stdout = fspk
        for tik, neu_ids in spks.items():
            for neu_id in neu_ids:
                for x,y,dedr_id,neu_idx in find_dedr_id(neu_id, axon_lines, 24):
                    print(tik, "spike", x, y, dedr_id, neu_idx)
        sys.stdout = __stdout__

def gen_spk_dwnc(fspikes, in_axon, dwnc = None, steps = None):
    if fspikes is not None:
        start_time = time.time_ns()
        print("===<0>=== generating spk.dwnc")
        if steps is None:
            if dwnc is None:
                steps = 0
            else:
                with open(dwnc,'r') as f:
                    steps = int(f.readlines()[-1].split()[0])
        with open(in_axon,'r') as f:
            axon_lines = f.readlines()
        with open(fspikes,'rb') as f:
            _spikes = pickle.load(f)
        spikes = {}
        for i in range(steps):
            if i in _spikes.keys():
                spikes[i] = _spikes[i]
        gen_spike(spikes, axon_lines)
        end_time = time.time_ns()
        print('===<0>=== gen_spk_dwnc elapsed : %.3f ms' % ((end_time - start_time)/1000000))

def read_spk_in(fn='stimulus.txt'):
    start_time = time.time_ns()
    print("===<0>=== generating spk.dwnc")
    if os.path.exists(fn):
        with open(fn,'r') as in_f:
            spk_in = open('spk.dwnc','w')
            __stdout__ = sys.stdout
            sys.stdout = spk_in
            tmp = in_f.readline()
            while '#' in tmp:
                tmp = in_f.readline()
            tiks = tmp.split()
            tot_tik = int(tiks[-1])
            print("# tot_tik: %d" % tot_tik)
            index = 0
            while(index < tot_tik):
                items = in_f.readline()
                item = items.split()
                if len(item) < 1:
                    continue
                if '#' in item[0]:
                    continue
                spk_num = int(item[0])
                for i in range(0,spk_num):
                    items = in_f.readline()
                    item = items.split()
                    if len(item) < 1:
                        continue
                    if '#' in item[0]:
                        continue
                    if (item[3] == 'false'):
                        t = "spike"
                    else:
                        t = "reward"
                    print("%4d 2 %s  11  13 \"0x%04x\" \"0x%03x\"" % (index,t,int(item[2]),int(item[4])))
                index += 1
            sys.stdout = __stdout__
            spk_in.close()
        in_f.close()
    end_time = time.time_ns()
    print('===<0>=== read_spk_in elapsed : %.3f ms' % ((end_time - start_time)/1000000))

def read_spk_out(fn='out.h5',chip_bin='chip_0_0.bin'):
    start_time = time.time_ns()
    print("===<0>=== generating spk_out.dwnc")
    if not os.path.exists(chip_bin):
        return
    n = Node.Node(0,0,chip_bin)
    c = n.getRams()[Node.CBlock.CONF.value]
    if os.path.exists(fn):
        f = h5py.File(fn,"r")
        tik_div = f.attrs['frequency_division_coef']
        index = 1
        vc = 2
        spk_out = open('spk_out.dwnc','w')
        __stdout__ = sys.stdout
        sys.stdout = spk_out
        spk = 'spike'
        rwd = 'reward'
        if (c.getRamBit(Node.CConf.NM_WORK_MODE.value,5)):
            spk = 'spike_short'
            rwd = 'reward_short'
        if (c.getRamBit(Node.CConf.NM_WORK_MODE.value,6)):
            spk = rwd
        for pkgs in f['neuron']['spike_packet']:
            for pkg in pkgs:
                if(pkg[0] < 0):
                    pkg[0] = -16 - pkg[0]
                if(pkg[1] < 0):
                    pkg[1] = -16 - pkg[1]
                print("%4d %d %s %2d  %2d \"0x%04x\" \"0x%03x\" %2d %2d" % (index*tik_div,vc,spk,pkg[0],pkg[1],pkg[2],pkg[3],0,0))
                vc = vc << 1
                if (vc > 8):
                    vc = 1
            index += 1
        sys.stdout = __stdout__
        spk_out.close()
        f.close()
    end_time = time.time_ns()
    print('===<0>=== read_spk_out elapsed : %.3f ms' % ((end_time - start_time)/1000000))

def gen_flit(item,fin,fbin,direct=0,x_from=-1,y_from=-1):
    global last_vc
    global tick
    global start_tick
    global stop_tick
    global clear_tick
    global pkg_num
    if direct == 0:
        pkg_num += 1
    tik  = int(item[0])
    cmd  = "0xc0000000"
    if (item[1] == "cmd"):
        cmd = item[2]
    while(isinstance(cmd,str)):
        cmd = eval(cmd)
    cmd = cmd >> 24
    if tik != tick  and tik > 0 and (item[1] != 'cmd' or cmd != 0b11011000):
        cmd = 0b011000
        arg = (tik - tick - 1)&0xffffff
        cmd_f = 0x3
        if direct == 0:
            l = (cmd_f<<30)+(cmd<<24)+arg
            ss_l = b"%08x\n" % l
            fin.write(ss_l)
            fbin.write(struct.pack('I',l))
        else:
            l = (cmd_f<<30)+(cmd<<24)
            ss_l = b"%08x\n" % l
            for i in range(arg+1):
                fin.write(ss_l)
                fbin.write(struct.pack('I',l))
    if tik > 0:
        tick = tik
    #vc   = int(item[1])
    vc   = 0
    if (vc == 0):
        vc = last_vc << 1
        if (vc > 8):
            vc = 1
    elif (vc not in (1,2,4,8)):
        vc_list = []
        if (vc & 0x1):
            vc_list.append(1)
        if (vc & 0x2):
            vc_list.append(2)
        if (vc & 0x4):
            vc_list.append(4)
        if (vc & 0x8):
            vc_list.append(8)
        vc = random.choice(vc_list)
    last_vc = vc
    op   = item[1]
    cmd  = "0x80000000"
    if (op == "cmd"):
        cmd = item[2]
        x   = 0
        y   = 0
        x_src = 0
        x_dff = 0
        x_sig = 0
        y_src = 0
        y_dff = 0
        y_sig = 0
        if "0xc0000001" in cmd:
            start_tick = tik
        if "0xc0000000" in cmd:
            stop_tick = tik
        if "0xd0000000" in cmd:
            clear_tick = tik
            #if stop_tick == -1 or clear_tick != stop_tick:
            #    print("clear tick must follow stop tick in same step!")
            #    sys.exit(1)
    else:
        x    = int(item[2])
        y    = int(item[3])
        addr = item[4]
        if (len(item) > 5):
            data = item[5]
        else:
            data = 0
        if (op == "spike" or op == "spike_short" or op == "reward" or op == "reward_short" or op == "write" or op == "write_risc" or op == "read_ack" or op == "read_risc_ack" or op == "flow_ack" or op == "read"):
            if (len(item) > 6):
                x_from = int(item[6])
            if (len(item) > 7):
                y_from = int(item[7])
        if (op == "flow"):
            if (len(item) > 5):
                x_from = int(item[5])
            if (len(item) > 6):
                y_from = int(item[6])

        x_src = x_from - x
        if (x_src > 0):
            x_sig = 1
        else:
            x_src = -x_src
            x_sig = 0
        if (y_from != -1 and x_from != -1):
            if (x_src != 0):
                if (x_sig == 1):
                    x_dff = -1 - x
                else:
                    x_dff = 24 - x
            else:
                x_dff = 0
        elif (x_src >= 1):
            x_dff = x_src - 1
        else:
            x_dff = 0
        if (x_src == 16):
            x_src = 15
        if (y_from == -1):
            if (y < 0):
                y_sig = 1
                y_dff = -y-1
                y_src = -y
            elif (y < 24):
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
            if (y_src > 0):
                y_sig = 1
                y_dff = -1 - y
            elif (y_src < 0):
                y_src = -y_src
                y_dff = 24 - y
            else:
                y_dff = 0
            if (y_src == 16):
                y_src = 15

    if (x_dff > 0):
        if (x_sig == 1):
            port = "01000"
        else :
            port = "00010"
    else:
        if (y_dff == 0):
            port = "00001"
        elif (y_sig == 1):
            port = "00100"
        else:
            port = "10000"
    route_id = y
    if (y_from != -1):
        route_id = y_from
    else:
        if (y < 0):
            route_id = 0
        elif (y > 23):
            route_id = 23

    if (op == "read_risc_ack"):
        if (x_dff == 0):
            x_sig = 1
        if (y_dff == 0):
            y_sig = 1

    direct   = 2
    cmd_tmp = cmd
    while(isinstance(cmd_tmp,str)):
        cmd_tmp = eval(cmd_tmp)
    pclass   = op
    vcnum    = (route_id & 0xf) + (direct << 6)
    vcnum2   = (direct << 6)
    if direct == 1:
        vcnum2   = (direct << 6)
    port     = eval("0b"+port)
    port     = onehot2bin(port)
    if(pclass == "cmd") :
        while isinstance(cmd,str):
            cmd    = eval(cmd)
        l = cmd 
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "write_risc" or pclass == "read_risc_ack") :
        while isinstance(addr,str):
            addr   = eval(addr)
        while isinstance(data,str):
            data   = eval(data)
        if (pclass == "read_risc_ack"):
            l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1) + (1 << 0)
        else:
            l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x0 << 30) + (addr << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x0 << 30) + ((data & 0xFFFF0000) >> 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + ((data & 0xFFFF) << 15)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "write" or pclass == "read_ack") :
        while isinstance(addr,str):
            addr   = eval(addr)
        while isinstance(data,str):
            data   = eval(data)
        if (pclass == "read_ack"):
            l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1) + (1 << 0)
        else:
            l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x0 << 30) + (addr << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x0 << 30) + ((data & 0xFFFFFF000000) >> 21)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + ((data & 0xFFFFFF) << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "read") :
        while isinstance(addr,str):
            addr   = eval(addr)
        l = (0x2 << 30) + (route_id << 25) + (0x2 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + (addr << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "flow") :
        data1 = addr
        while isinstance(data1,str):
            data1   = eval(data1)
        l = (0x2 << 30) + (route_id << 25) + (0x3 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + (data1 << 3) 
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "flow_ack") :
        data1 = addr
        while isinstance(data1,str):
            data1   = eval(data1)
        data2 = data
        while isinstance(data2,str):
            data2   = eval(data2)
        l = (0x2 << 30) + (route_id << 25) + (0x7 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x0 << 30) + (data1 << 3) 
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x0 << 30) + ((data2 & 0x3FFFFFF8000000) >> 24)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + ((data2 & 0x7FFFFFF) << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "spike") :
        dedr_id   = addr
        neu_idx   = data
        while isinstance(dedr_id,str):
            dedr_id  = eval(dedr_id)
        while isinstance(neu_idx,str):
            neu_idx   = eval(neu_idx)
        l = (0x2 << 30) + (route_id << 25) + (0x0 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "spike_short") :
        dedr_id   = addr
        neu_idx   = data
        while isinstance(dedr_id,str):
            dedr_id  = eval(dedr_id)
        while isinstance(neu_idx,str):
            neu_idx   = eval(neu_idx)
        l = (0x2 << 30) + (route_id << 25) + (0x4 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "reward") :
        dedr_id   = addr
        neu_idx   = data
        while isinstance(dedr_id,str):
            dedr_id  = eval(dedr_id)
        while isinstance(neu_idx,str):
            neu_idx   = eval(neu_idx)
        l = (0x2 << 30) + (route_id << 25) + (0x5 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
    if(pclass == "reward_short") :
        dedr_id   = addr
        neu_idx   = data
        while isinstance(dedr_id,str):
            dedr_id  = eval(dedr_id)
        while isinstance(neu_idx,str):
            neu_idx   = eval(neu_idx)
        l = (0x2 << 30) + (route_id << 25) + (0x6 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))
        l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        ss_l = b"%08x\n" % l
        fin.write(ss_l)
        fbin.write(struct.pack('I',l))


FLIT_TEXT_LENGTH_BYTE = 8
FLIT_TEXT_NUM_BYTE = 4
FLIT_TEXT_LENGTH = FLIT_TEXT_NUM_BYTE * (FLIT_TEXT_LENGTH_BYTE + 1)
FLIT_BINARY_LENGTH_VALUE = 4
FLIT_BINARY_NUM_VALUE = 4
FLIT_BINARY_LENGTH = FLIT_BINARY_NUM_VALUE * FLIT_BINARY_LENGTH_VALUE

#def gen_flit_parallel(item,fin,fbin,direct=0,x_from=-1,y_from=-1):
def gen_flit_parallel(x, y, address, value, text_buffer, text_offset, binary_buffer, binary_offset):
    # global last_vc
    # global tick
    # global start_tick
    # global stop_tick
    # global clear_tick
    # global pkg_num
    # if direct == 0:
        # pkg_num += 1
    # tik  = int(item[0])
    # cmd  = "0xc0000000"
    # if (item[1] == "cmd"):
        # cmd = item[2]
    # while(isinstance(cmd,str)):
        # cmd = eval(cmd)
    # cmd = cmd >> 24
    # if tik != tick and (item[1] != 'cmd' or cmd != 0b11011000):
        # cmd = 0b011000
        # arg = (tik - tick - 1)&0xffffff
        # cmd_f = 0x3
        # l = (cmd_f<<30)+(cmd<<24)+arg
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # tick = tik
    #vc   = int(item[1])
    # vc   = 0
    # if (vc == 0):
        # vc = last_vc << 1
        # if (vc > 8):
            # vc = 1
    # elif (vc not in (1,2,4,8)):
        # vc_list = []
        # if (vc & 0x1):
            # vc_list.append(1)
        # if (vc & 0x2):
            # vc_list.append(2)
        # if (vc & 0x4):
            # vc_list.append(4)
        # if (vc & 0x8):
            # vc_list.append(8)
        # vc = random.choice(vc_list)
    # last_vc = vc
    # op   = item[1]
    # cmd  = "0x80000000"
    # if (op == "cmd"):
        # cmd = item[2]
        # x   = 0
        # y   = 0
        # x_src = 0
        # x_dff = 0
        # x_sig = 0
        # y_src = 0
        # y_dff = 0
        # y_sig = 0
        # # if "0xc0000001" in cmd:
            # # start_tick = tik
        # # if "0xc0000000" in cmd:
            # # stop_tick = tik
        # # if "0xd0000000" in cmd:
            # # clear_tick = tik
            # # if stop_tick == -1 or clear_tick != stop_tick:
                # # print("clear tick must follow stop tick in same step!")
                # # sys.exit(1)
    # else:
    # x    = int(item[2])
    # y    = int(item[3])
    # addr = item[4]
    # if (len(item) > 5):
        # data = item[5]
    # else:
        # data = 0
    # if (op == "spike" or op == "spike_short" or op == "reward" or op == "reward_short" or op == "write" or op == "write_risc" or op == "read_ack" or op == "read_risc_ack" or op == "flow_ack" or op == "read"):
        # if (len(item) > 6):
            # x_from = int(item[6])
        # if (len(item) > 7):
            # y_from = int(item[7])
    # if (op == "flow"):
        # if (len(item) > 5):
            # x_from = int(item[5])
        # if (len(item) > 6):
            # y_from = int(item[6])

    # x_src = x_from - x
    # if (x_src > 0):
        # x_sig = 1
    # else:
        # x_src = -x_src
        # x_sig = 0
    # if (y_from != -1 and x_from != -1):
        # if (x_src != 0):
            # if (x_sig == 1):
                # x_dff = -1 - x
            # else:
                # x_dff = 24 - x
        # else:
            # x_dff = 0
    # elif (x_src >= 1):
        # x_dff = x_src - 1
    # else:
        # x_dff = 0
    # if (x_src == 16):
        # x_src = 15
    # if (y_from == -1):
        # if (y < 0):
            # y_sig = 1
            # y_dff = -y-1
            # y_src = -y
        # elif (y < 24):
            # y_sig = 0
            # y_dff = 0
            # y_src = 0
        # else:
            # y_sig = 0
            # y_dff = y - 24
            # y_src = y - 23
    # else:
        # y_sig = 0
        # y_src = y_from - y
        # if (y_src > 0):
            # y_sig = 1
            # y_dff = -1 - y
        # elif (y_src < 0):
            # y_src = -y_src
            # y_dff = 24 - y
        # else:
            # y_dff = 0
        # if (y_src == 16):
            # y_src = 15
    src_x = 24 - x
    src_y = 0
    diff_x = 23 - x
    diff_y = 0
    sign_x = 1
    sign_y = 0

    if (diff_x > 0):
        # if (x_sig == 1):
            # port = "01000"
        # else :
        port = 3
    else:
        # if (y_dff == 0):
        port = 0
        # elif (y_sig == 1):
            # port = "00100"
        # else:
            # port = "10000"
    route_id = y
    # if (y_from != -1):
    #   route_id = y_from
    # else:
        # if (y < 0):
            # route_id = 0
        # elif (y > 23):
            # route_id = 23

    # if (op == "read_risc_ack"):
        # if (diff_x == 0):
            # x_sig = 1
        # if (y_dff == 0):
            # y_sig = 1

    direct   = 2
    # cmd_tmp = cmd
    # while(isinstance(cmd_tmp,str)):
        # cmd_tmp = eval(cmd_tmp)
    # pclass   = op
    vcnum    = (route_id & 0xf) + (direct << 6)
    vcnum2   = (direct << 6)
    # if direct == 1:
        # vcnum2   = (direct << 6)
    msb      = route_id >> 4
    #port     = eval("0b"+port)
    #port     = onehot2bin(port) + (msb << 4)
    #port |= route_id >> 4
    # if(pclass == "cmd") :
        # while isinstance(cmd,str):
            # cmd    = eval(cmd)
        # l = cmd 
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "write_risc" or pclass == "read_risc_ack") :
        # while isinstance(addr,str):
            # addr   = eval(addr)
        # while isinstance(data,str):
            # data   = eval(data)
        # if (pclass == "read_risc_ack"):
            # l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1) + (1 << 0)
        # else:
            # l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x0 << 30) + (addr << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x0 << 30) + ((data & 0xFFFF0000) >> 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + ((data & 0xFFFF) << 15)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "write" or pclass == "read_ack") :
    # while isinstance(addr,str):
        # addr   = eval(addr)
    # while isinstance(data,str):
        # data   = eval(data)
    # if (pclass == "read_ack"):
        # l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1) + (1 << 0)
    # else:
    #l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (x_sig << 18)+ (diff_x << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
    #l = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (diff_x << 14) + (src_x << 5)
    #ss_l = "%08x" % l
    #fin.write(ss_l + '\n')
    #fbin.write(struct.pack('I',l))
    #l = (0x0 << 30) + (address << 3)
    #ss_l = "%08x" % l
    #fin.write(ss_l + '\n')
    #fbin.write(struct.pack('I',l))
    #l = (0x0 << 30) + ((value & 0xFFFFFF000000) >> 21)
    #ss_l = "%08x" % l
    #fin.write(ss_l + '\n')
    #fbin.write(struct.pack('I',l))
    #l = (0x1 << 30) + ((value & 0xFFFFFF) << 3)
    #ss_l = "%08x" % l
    #fin.write(ss_l + '\n')
    #fbin.write(struct.pack('I',l))
    head = (0x2 << 30) + (route_id << 25) + (0x1 << 22) + (port << 19) + (sign_x << 18) + (diff_x << 14) + (src_x << 5)
    body0 = (0x0 << 30) + (address << 3)
    body1 = (0x0 << 30) + ((value & 0xFFFFFF000000) >> 21)
    tail = (0x1 << 30) + ((value & 0xFFFFFF) << 3)
    text_buffer[text_offset : text_offset + FLIT_TEXT_LENGTH] = \
        b"%08x\n%08x\n%08x\n%08x\n" % (head, body0, body1, tail)
    struct.pack_into("<4I", binary_buffer, binary_offset, head, body0, body1, tail)
    # if(pclass == "read") :
        # while isinstance(addr,str):
            # addr   = eval(addr)
        # l = (0x2 << 30) + (route_id << 25) + (0x2 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + (addr << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "flow") :
        # data1 = addr
        # while isinstance(data1,str):
            # data1   = eval(data1)
        # l = (0x2 << 30) + (route_id << 25) + (0x3 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + (data1 << 3) 
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "flow_ack") :
        # data1 = addr
        # while isinstance(data1,str):
            # data1   = eval(data1)
        # data2 = data
        # while isinstance(data2,str):
            # data2   = eval(data2)
        # l = (0x2 << 30) + (route_id << 25) + (0x7 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x0 << 30) + (data1 << 3) 
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x0 << 30) + ((data2 & 0x3FFFFFF8000000) >> 24)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + ((data2 & 0x7FFFFFF) << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "spike") :
        # dedr_id   = addr
        # neu_idx   = data
        # while isinstance(dedr_id,str):
            # dedr_id  = eval(dedr_id)
        # while isinstance(neu_idx,str):
            # neu_idx   = eval(neu_idx)
        # l = (0x2 << 30) + (route_id << 25) + (0x0 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "spike_short") :
        # dedr_id   = addr
        # neu_idx   = data
        # while isinstance(dedr_id,str):
            # dedr_id  = eval(dedr_id)
        # while isinstance(neu_idx,str):
            # neu_idx   = eval(neu_idx)
        # l = (0x2 << 30) + (route_id << 25) + (0x4 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "reward") :
        # dedr_id   = addr
        # neu_idx   = data
        # while isinstance(dedr_id,str):
            # dedr_id  = eval(dedr_id)
        # while isinstance(neu_idx,str):
            # neu_idx   = eval(neu_idx)
        # l = (0x2 << 30) + (route_id << 25) + (0x5 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
    # if(pclass == "reward_short") :
        # dedr_id   = addr
        # neu_idx   = data
        # while isinstance(dedr_id,str):
            # dedr_id  = eval(dedr_id)
        # while isinstance(neu_idx,str):
            # neu_idx   = eval(neu_idx)
        # l = (0x2 << 30) + (route_id << 25) + (0x6 << 22) + (port << 19) + (x_sig << 18)+ (x_dff << 14) + (y_sig << 13) + (y_dff << 9)+ (x_src << 5) + (y_src << 1)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))
        # l = (0x1 << 30) + (dedr_id << 15) + (neu_idx << 3)
        # ss_l = "%08x" % l
        # fin.write(ss_l + '\n')
        # fbin.write(struct.pack('I',l))

def gen_flit_by_fn(fn,fin,fbin,direct=0,tc=""):
    if not os.path.exists(fn):
        return
    with open(fn, 'r', encoding='utf-8') as load_f :
        lines = load_f.readlines()
        for items in lines:
            item = items.split()
            if len(item) < 2:
                continue
            if '#' in item[0]:
                continue
            if (item[0] == "<<"):
                gen_flit_by_fn(item[1],fin,fbin,direct,tc)
            elif (item[1] == "read" and len(item) >= 6):
                tmp = item
                addr = eval(eval(item[4]))
                while(isinstance(item[5],str)):
                    item[5]=eval(item[5])
                for i in range(int(item[5])):
                    tmp[4] = "\"%s\"" % hex(addr+i)
                    gen_flit(tmp,fin,fbin,direct,24)
            elif ((item[1] == "write" or items[1] == "write_ram" or item[1] == "read_ack" or item[1] == "write_risc" or item[1] == "read_risc_ack") and len(item) == 5):
                if item[1] == "write_ram":
                    item[1] = "write"
                tmp = item
                tmp.append("")
                if os.path.exists(tc+item[4]):
                    with open(tc+item[4],'rb') as write_f:
                        tot = int.from_bytes(write_f.read(4),byteorder='little',signed=False)
                        for segment in range(tot):
                            area_id = int(write_f.read(1)[0])
                            t = (area_id & 0xF)
                            config_word_equal = ((t & 0x80) != 0)
                            addr = int.from_bytes(write_f.read(4),byteorder='little',signed=False)
                            length = int.from_bytes(write_f.read(4),byteorder='little',signed=False)
                            bo = 'little'
                            bs = 4
                            di = 1
                            # Dedr
                            if (t == 0x01):
                                addr += 0x10000
                                bs = 6
                            # Wgtsum
                            if (t == 0x02):
                                addr += 0x4000
                                bs = 2
                            # Inference State
                            if (t == 0x03):
                                addr += 0x10000
                                bs = 6
                                di = -1 
                            # Voltage
                            if (t == 0x04):
                                addr += 0x2000
                                bs = 2
                            # Axon
                            if (t == 0x05):
                                addr += 0x8000
                                bs = 4
                            # Learn State
                            if (t == 0x06):
                                addr += 0x10000
                                bs = 6
                                di = -1
                            # Inst
                            if (t == 0x07):
                                addr += 0x1000
                                bs = 2
                            # Reg
                            if (t == 0x08):
                                addr += 0x800
                                bs = 2
                            size = length if config_word_equal else int(length / bs)
                            x, y = int(item[2]), int(item[3])
                            address = np.arange(addr, addr + size * di, di)
                            if config_word_equal:
                                #tmp[5] = "\"0x%x\"" % int.from_bytes(write_f.read(bs),byteorder=bo,signed=False)
                                value = struct.unpack_from("<Q", write_f.read(bs) + b'\x00' * (8 - bs))[0]
                                text_buffer = bytes(FLIT_TEXT_LENGTH)
                                binary_buffer = bytes(FLIT_BINARY_LENGTH)
                                gen_flit_parallel(x, y, address, value, text_buffer, 0, binary_buffer, 0)
                                text_buffer = text_buffer * size
                                binary_buffer = binary_buffer * size
                                # for i in range(length):
                                    # tmp[4] = "\"%s\"" % hex(addr)
                                    # #tmp[5] = "\"0x%x\"" % int.from_bytes(write_f.read(bs),byteorder=bo,signed=False)
                                    # gen_flit(tmp,fin,fbin,direct)
                                    # #no_end = segment < tot-1
                                    # addr = addr + di
                            else:
                                buffer = write_f.read(length)
                                index_byte = np.arange(0, length, bs)
                                text_buffer = bytearray(size * FLIT_TEXT_LENGTH)
                                text_offset = np.arange(0, size * FLIT_TEXT_LENGTH, FLIT_TEXT_LENGTH)
                                binary_buffer = bytearray(size * FLIT_BINARY_LENGTH)
                                binary_offset = np.arange(0, size * FLIT_BINARY_LENGTH, FLIT_BINARY_LENGTH)
                                # for i in range(size):
                                    # tmp[4] = "\"%s\"" % hex(addr)
                                    # tmp[5] = "\"0x%x\"" % int.from_bytes(write_f.read(bs),byteorder=bo,signed=False)
                                    # gen_flit(tmp,fin,fbin,direct)
                                    # #no_end = segment < tot-1
                                    # addr = addr + di
                                def convert(x, y, address, index_byte, text_offset, binary_offset):
                                    buffer_value = buffer[index_byte : index_byte + bs] + b'\x00' * (8 - bs)
                                    value = struct.unpack("<Q", buffer_value)[0]
                                    gen_flit_parallel(
                                        x,
                                        y,
                                        address,
                                        value,
                                        text_buffer,
                                        text_offset,
                                        binary_buffer,
                                        binary_offset)
                                np.frompyfunc(convert, 6, 0)(
                                    x,
                                    y,
                                    address,
                                    index_byte,
                                    text_offset,
                                    binary_offset)
                            fin.write(text_buffer)
                            fbin.write(binary_buffer)
                    #write_f.close()
            elif ((item[1] == "write" or item[1] == "write_ram" or item[1] == "read_ack" or item[1] == "write_risc" or item[1] == "read_risc_ack") and os.path.exists(tc+item[5])):
                if item[1] == "write_ram":
                    item[1] = "write"
                tmp = item
                x, y = int(item[2]), int(item[3])
                addr = item[4]
                while isinstance(addr,str):
                    addr = eval(addr)
                with open(tc+item[5], 'r') as write_f :
                    wlines = write_f.readlines()
                    wlength = len(wlines)
                    #windex = 0
                    # for witems in wlines:
                        # windex  = windex + 1
                        # tmp[5] = "\"0x%s\"" % witems.replace("\n","")
                        # tmp[4] = "\"%s\"" % hex(addr)
                        # gen_flit(tmp,fin,fbin,direct)
                        # addr = addr + 1
                    address = np.arange(addr, addr + wlength)
                    text_buffer = bytearray(wlength * FLIT_TEXT_LENGTH)
                    text_offset = np.arange(0, wlength * FLIT_TEXT_LENGTH, FLIT_TEXT_LENGTH)
                    binary_buffer = bytearray(wlength * FLIT_BINARY_LENGTH)
                    binary_offset = np.arange(0, wlength * FLIT_BINARY_LENGTH, FLIT_BINARY_LENGTH)
                    def convert(x, y, address, line, text_offset, binary_offset):
                        return \
                            gen_flit_parallel(
                                x,
                                y,
                                address,
                                int(line, 16),
                                text_buffer,
                                text_offset,
                                binary_buffer,
                                binary_offset)
                    np.frompyfunc(convert, 6, 0)(
                        x,
                        y,
                        address,
                        wlines,
                        text_offset,
                        binary_offset)
                    fin.write(text_buffer)
                    fbin.write(binary_buffer)
                #write_f.close()
            else:
                gen_flit(item,fin,fbin,direct,24)
    load_f.close()

def flit_gen(tc="", pre="", dwnc = "input.dwnc", dwnc_out = 'output.dwnc', update=1):
    global tick
    start_time = time.time_ns()
    print("===<1>=== generating %sflitin.txt" % pre)
    #os.system("dwnc_ml")
    #pre=os.path.basename(os.getcwd()) + "_"

    if tc != "" :
       if os.path.exists(tc):
           tc  = tc+"/"
           pre = tc + "_"
       else:
           tc = ""
    if os.path.isfile(tc+pre+dwnc) :
       need_update = 1
       if (os.path.exists(tc+pre+'flitin.txt')):
           in_time = time.ctime(os.path.getmtime(tc+pre+dwnc))
           flit_time = time.ctime(os.path.getmtime(tc+pre+'flitin.txt'))
           if (flit_time > in_time):
               need_update = update
       if (need_update == 0):
           print('===<1>=== flit_gen skipped' )
           return
       fin   = open(tc+pre+'flitin.txt'  , "wb")
       fin_bin  = open(tc+pre+'flitin.bin'  , "wb")
       tick = 0
       gen_flit_by_fn(tc+pre+dwnc,fin,fin_bin,0,tc)
       fin.close()
       fin_bin.close()
       
       fout   = open(tc+pre+'flitout.txt'  , "wb")
       fout_bin = open(tc+pre+'flitout.bin'  , "wb")
       tick = 0
       gen_flit_by_fn(tc+pre+dwnc_out,fout,fout_bin,1,tc)
       fout.close()
       fout_bin.close()
       
    end_time = time.time_ns()
    print('===<1>=== flit_gen elapsed : %.3f ms' % ((end_time - start_time)/1000000))

def run_dwnc(tc="", pre="", recv = True, ip = "10.11.8.213", port = 7):
    if tc != "" :
        if os.path.exists(tc):
            tc  = tc+"/"
        else:
            tc = ""
    trans = Transmitter()
    ip_address = (ip,port)
    trans.connect_lwip(ip_address)
    print("===<2>=== tcp connect succeed")
    start_time = time.time_ns()
    #res=trans.send_flit(tc+pre+"flitin.txt")
    res=trans.send_flit_bin(tc+pre+"flitin.bin")
    if res == 0:
        return
    end_time = time.time_ns()
    print("===<2>=== send flit data   succeed")
    print('===<2>=== tcp sent elapsed : %.3f ms' % ((end_time - start_time)/1000000))

    f = open(tc+"recv_"+pre+"flitout.txt", "wb")
    fbin = open(tc+"recv_"+pre+"flitout.bin", "wb")
    start_time = time.time_ns()
    hl = b""
    index = 0
    tot = 0
    while recv:
        request = trans.socket_inst.recv(1024)
        if len(request) <= 0:
            break
        fbin.write(request)
        for i in range(len(request)):
            b = b"%02x" % request[i]
            hl = b + hl
            index = index + 1
            if (index == 4):
                f.write (hl + b"\n")
                # print(hl)
                hl = b""
                index = 0
                tot = tot + 1         
    end_time = time.time_ns()
    f.close()
    fbin.close()
    trans.socket_inst.close()
    print('===<2>=== tcp recv elapsed : %.3f ms with %d flits' % ((end_time - start_time)/1000000,tot))

def run_parser(recv_flitout='recv_flitout.txt'):
    print('===<3>=== parser recv flit : %s' % recv_flitout)
    start_time = time.time_ns()
    with open(recv_flitout, 'r') as flit_f :
        lines = flit_f.readlines()
        index = 0
        is_write = 0
        is_spike = 0
        is_reward = 0
        is_flow = 0
        is_dedr = 0
        t = 0
        for items in lines:
            if (eval("0x"+items[0])>>2)&0x3 == 3 :
                cmd = eval("0x"+items[0:2])&0x3f
                if cmd == 0b011000:
                    arg = eval("0x"+items[2:8])
                    t+=arg+1
            elif (eval("0x"+items[0])>>2)&0x3 == 2 :
                index = 0
                is_write = (eval("0x"+items[1:3])>>2) & 0x7 == 1
                is_spike = (eval("0x"+items[1:3])>>2) & 0x7 in (0,4,5,6)
                is_reward= (eval("0x"+items[1:3])>>2) & 0x7 in (5,6)
                is_flow  = (eval("0x"+items[1:3])>>2) & 0x7 == 7
                y = (eval("0x"+items[0:2])>>1)&0x1f
                x = 24 - ((eval("0x"+items[5:7])>>1)&0xf)
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
                if (eval("0x"+items[0])>>2)&0x3 == 1 :
                    neu_idx = (eval("0x"+items[4:8]) >> 3) & 0xfff
                    dedr_id = (eval("0x"+items[0:5]) >> 3) & 0x7fff                    
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
    flit_f.close()
    end_time = time.time_ns()
    print('===<3>=== parser recv flit elapsed : %.3f ms' % ((end_time - start_time)/1000000))

def run_app():
    options, _ = arg_parse()
    stime   = time.time_ns()
    dwnc    = options.din
    dwnc_out= options.dout
    ip_table= {'ALG':'10.11.8.213','OS':'10.11.8.4'}
    app     = options.app
    pre     = options.prefix
    analyse = options.analyse
    group   = options.group
    ip      = options.ip
    port    = options.port
    in_axon = options.input_axon
    fspikes = options.spikes
    steps   = options.steps
    fconn   = options.connections
    rsm     = options.rsm
    gt      = options.gt
    vth     = options.vth
    skip_c  = options.skip_compile
    skip_r  = options.skip_run
    if group is not None:
        ip = ip_table[group]
    if pre != "" and not options.no_underscore:
        pre += "_"
    if not os.path.exists(app):
        app = ""
    if app == "":
        print('\n[app is %s] [dwnc is %s]' % (os.path.basename(os.getcwd()),pre+dwnc))
    else:
        print('\n[app is %s] [dwnc is %s]' % (app,pre+dwnc))
    if (app != ""):
        os.chdir(app)
    gen_config_dwnc(fconn, rsm=rsm, gt=gt, vth=vth)
    gen_spk_dwnc(fspikes, in_axon, pre+dwnc, steps)
    if not skip_c:
        flit_gen("",pre,dwnc,dwnc_out)
        print("pkg_num: %d, start: %d, stop: %d, clear: %d" % (pkg_num, start_tick, stop_tick, clear_tick))
        if (start_tick == -1 or stop_tick == -1 or start_tick > stop_tick) and pkg_num >0:
            if start_tick == -1:
                print("missing start tick!")
            elif stop_tick == -1:
                print("missing stop tick!")
            elif start_tick > stop_tick:
                print("stop tick should not be ahead of start tick!")
            sys.exit(1)
    if not skip_r:
        run_dwnc("",pre=pre,ip=ip,port=port)
    if analyse:
        run_parser("recv_" + pre + "flitout.txt")
    etime = time.time_ns()
    print("[done in %.3f ms]" % ((etime - stime)/1000000.0))

def arg_parse():
    parser = argparse.ArgumentParser(prog='run_dwnc32_east', description='Run darwin code on Darwin III by east')
    parser.add_argument('-app', type=str, default='', help='path of the darwin code')
    parser.add_argument('-prefix', type=str, default='', help=r'prefix of the darwin code, ${prefix}_input.dwnc will be executed')
    parser.add_argument('-connections', type=str, help='connections in pickle to generate config.dwnc')
    parser.add_argument('-rsm', type=int, choices=[0,1,2,3], default=2, help='reset mode of neurons, 0:no reset;1:to zero;2:to vt-vth;3:to vt-vt_dec')
    parser.add_argument('-gt', action='store_true', default=False, help='use vt > vth to generate spike instead of vt >= vth')
    parser.add_argument('-vth', type=int, default=0, help='threshold of neurons')
    parser.add_argument('-spikes', type=str, help='spikes in pickle to generate spikes.dwnc')
    parser.add_argument('-input_axon', type=str, default='24-13-ax.txt', help='axon of the input node (default: %(default)s)')
    parser.add_argument('-steps', type=int, help='time steps to run')
    parser.add_argument('-analyse', action='store_true', default=False, help='analyse the data received from Darwin III')
    parser.add_argument('-skip_compile', action='store_true', default=False, help='skip compiling darwin code')
    parser.add_argument('-skip_run', action='store_true', default=False, help='skip running flit on Darwin III')
    parser.add_argument('-din', type=str, default='input.dwnc', help=argparse.SUPPRESS)
    parser.add_argument('-dout', type=str, default='output.dwnc', help=argparse.SUPPRESS)
    parser.add_argument('-no_underscore', action='store_true', default=False, help=argparse.SUPPRESS)
    parser.add_argument('-port', type=int, default=7, help=argparse.SUPPRESS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-group', type=str, choices=['ALG','OS'], help='group of Darwin III')
    group.add_argument('-ip', type=str, default='10.11.8.213', help=argparse.SUPPRESS)

    return parser.parse_known_args()

if __name__ == "__main__":
    stime = time.time_ns()
    run_app()
    etime = time.time_ns()
    print("\n<--total time elapsed : %.3f ms-->\n" % ((etime - stime)/1000000.0))
