import socket
import sys
import struct
import os
import time
import mmap
import fcntl
import threading
import ctypes

s1 = threading.Semaphore(1)
receive_count : int = 0
active_ip = ""
steps  = 1
dma_id = 0

CHIP_RESET = 10

is_A9=os.system("lscpu | grep 'Model name' | awk '{print $NF}' | grep 'Cortex-A9$' > /dev/null")
if is_A9 == 0:
    FINISH_XFER = ord('a') + (ord('a') << 8) + (4 << 16) + (1 << 30)
    START_XFER  = ord('b') + (ord('a') << 8) + (4 << 16) + (1 << 30)
    XFER        = ord('c') + (ord('a') << 8) + (4 << 16) + (2 << 30)
    DMA_BUF_MAX = 4*1024*1024
    DMA_RES     = DMA_BUF_MAX
    DMA_LENGTH  = DMA_RES + 4
    DMA_MAP_SIZE= DMA_BUF_MAX + 8
    DMA_BINDEX  = b'\x00\x00\x00\x00'
    DMA_RX_LEN  = b'\x00\x10\x00\x00'
    RECV_SIZE   = 4194304
else:
    FINISH_XFER = ord('a') + (ord('a') << 8) + (8 << 16) + (1 << 30)
    START_XFER  = ord('b') + (ord('a') << 8) + (8 << 16) + (1 << 30)
    XFER        = ord('c') + (ord('a') << 8) + (8 << 16) + (2 << 30)
    DMA_BUF_MAX = 64*1024*1024
    DMA_RES     = DMA_BUF_MAX
    DMA_LENGTH  = DMA_RES + 4
    DMA_MAP_SIZE= DMA_BUF_MAX + 8
    DMA_BINDEX  = b'\x00\x00\x00\x00'
    DMA_RX_LEN  = b'\x00\x10\x00\x00'
    RECV_SIZE   = 8388608


class DMA_Transmitter(object):
    def __init__(self, id=0):
        self.id = id
        if self.id >= 2:
            self.id = self.id + 1
                                                                           
    def open(self):
        if self.id == 0:                                           
            self.tx_dma_fd = os.open("/dev/dma_proxy_tx", os.O_RDWR)           
            self.rx_dma_fd = os.open("/dev/dma_proxy_rx", os.O_RDWR)
        else:
            self.tx_dma_fd = os.open("/dev/dma_proxy_tx%d" % self.id, os.O_RDWR)           
            self.rx_dma_fd = os.open("/dev/dma_proxy_rx%d" % self.id, os.O_RDWR)           
                                                                           
    def close(self):                                                       
        os.close(self.tx_dma_fd)                                           
        os.close(self.rx_dma_fd)                                           
                                                                           
    def send_flit_bin(self, flit_bin):
        '''                                                                
        发送flit                                                           
        '''                                                                
        length = len(flit_bin)                                             
        if length > DMA_BUF_MAX:                                           
            print("===<2>=== %s is larger than %dMB" % (flit_bin_file,DMA_BUF_MAX>>20))
            print("===<2>=== send flit length failed")                                 
            return 0                                                                   
        #print("flit length: %d" % length)                                 
        with mmap.mmap(self.tx_dma_fd,DMA_MAP_SIZE,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ) as mm:
            mm[DMA_LENGTH:DMA_LENGTH+4]=struct.pack('I',length)                                            
            mm[:length]=flit_bin                                                                           
            fcntl.ioctl(self.tx_dma_fd, XFER, DMA_BINDEX)                                                  
            tx_result = struct.unpack('I',mm[DMA_RES:DMA_RES+4])[0]                                        
            if tx_result != 0:                                                                             
                return 0                                                                                   
        return 1 

    def recv_flit_bin(self, steps = 1):
        '''                                                                                                
        接收flit                                                                                           
        '''                                                                                                
        recv = bytearray()                                                                                       
        with mmap.mmap(self.rx_dma_fd,DMA_MAP_SIZE,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ) as mm:
            for i in range(steps):
                last = False                                                                                   
                while not last:                                                                                
                    mm[DMA_LENGTH:DMA_LENGTH+4]=DMA_RX_LEN                                                     
                    fcntl.ioctl(self.rx_dma_fd, XFER, DMA_BINDEX)                                              
                    rx_result = struct.unpack('I',mm[DMA_RES:DMA_RES+4])[0]                                    
                    #print("rx_result is %d" % rx_result)
                    if rx_result == 0:                                                                         
                        rx_length = struct.unpack('I',mm[DMA_LENGTH:DMA_LENGTH+4])[0]                          
                        last_flit = struct.unpack('I',mm[rx_length-4:rx_length])[0]                            
                        last = last_flit == 0xffffffff                                                         
                        if last:                                                                           
                            rx_length -= 4
                        #print("rx_length is %d" % rx_length)
                        recv+=mm[:rx_length]
                    else:
                        last = True
                else:                                                                                          
                    last = True
        #print("recv flit bytes: %d" % len(recv))
        return recv 

def recv(trans):
    s1.acquire()
    hl = b""
    index = 0
    with open("flitout.txt","wb") as f:
        flits = trans.recv_flit_bin(steps)
        for i in range(len(flits)):
            b = b"%02x" % flits[i]
            hl = b + hl
            index = index + 1
            if (index == 4):
                f.write (hl + b"\n")
                hl = b""
                index = 0
    s1.release()
    return flits

def send_and_recv(flit_file):
    with open(flit_file,"rb") as f:
        flit_bin = f.read()
        trans = DMA_Transmitter(dma_id)
        trans.open()
        if steps > 0:
            thread = threading.Thread(target=recv, args=(trans,))
            thread.start()
        trans.send_flit_bin(flit_bin)
        s1.acquire()
        s1.release()

if __name__=='__main__':
    flit_file = "flitin.bin"
    if len(sys.argv) > 1:
        flit_file = sys.argv[1]
    if len(sys.argv) > 2:
        dma_id = int(sys.argv[2])
    if len(sys.argv) > 3:
        steps = int(sys.argv[3])
    send_and_recv(flit_file)
