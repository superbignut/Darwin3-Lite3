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
steps = 1

CHIP_RESET = 10

_IOC_WRITE         = 1
_IOC_READ          = 2

AXIDMA_IOCTL_MAGIC = ord('W')
AXIDMA_NUM_IOCTLS  = 10

AXIDMA_GET_NUM_DMA_CHANNELS = (_IOC_WRITE << 30) + (20 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 0
AXIDMA_GET_DMA_CHANNELS     = (_IOC_READ  << 30) + (4  << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 1
AXIDMA_SET_DMA_SIGNAL       =                                   (AXIDMA_IOCTL_MAGIC << 8) + 2
AXIDMA_REGISTER_BUFFER      = (_IOC_READ  << 30) + (12 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 3
AXIDMA_DMA_READ             = (_IOC_READ  << 30) + (28 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 4
AXIDMA_DMA_WRITE            = (_IOC_READ  << 30) + (28 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 5
AXIDMA_DMA_READWRITE        = (_IOC_READ  << 30) + (30 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 6
AXIDMA_DMA_VIDEO_READ       = (_IOC_READ  << 30) + (24 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 7
AXIDMA_DMA_VIDEO_WRITE      = (_IOC_WRITE << 30) + (24 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 7
AXIDMA_STOP_DMA_CHANNEL     = (_IOC_READ  << 30) + (20 << 16) + (AXIDMA_IOCTL_MAGIC << 8) + 9
XIDMA_UNREGISTER_BUFFER     =                                   (AXIDMA_IOCTL_MAGIC << 8) + 10

DMA_RX_LEN  = b'\x00\x10\x00\x00'

is_A9=os.system("lscpu | grep 'Model name' | awk '{print $NF}' | grep 'Cortex-A9$' > /dev/null")
if is_A9 == 0:
    DMA_BUF_MAX = 4*1024*1024
    RECV_SIZE   = 4194304
else:
    DMA_BUF_MAX = 64*1024*1024
    RECV_SIZE   = 8388608

def bytes2ptr(bytes, pack=True):
    ptr4 = ctypes.POINTER(ctypes.c_uint32)(ctypes.c_uint32.from_buffer(bytes))
    ret  = ctypes.cast(ptr4, ctypes.c_void_p).value
    if pack:
        ret = struct.pack('I', ret)
    return ret

class struct_axidma_transaction:
    size  = 7*4
    bytes = bytearray(size)
    def __init__(self, bytes = None):
        if bytes is None:
            self.wait       = 0
            self.channel_id = 0
            self.buf        = 0
            self.buf_len    = 0
            self.height     = 0
            self.width      = 0
            self.depth      = 0
        else:
            self.set(bytes)

    def set(self, bytes):
        data = struct.unpack_from("7I",bytes)
        self.wait       = data[0]
        self.channel_id = data[1]
        self.buf        = data[2]
        self.buf_len    = data[3]
        self.height     = data[4]
        self.width      = data[5]
        self.depth      = data[6]

    def get(self):
        data = (self.wait, self.channel_id, self.buf, self.buf_len, self.height, self.width, self.depth)
        self.bytes[:] = struct.pack("7I", *data)
        return self.bytes

class DMA_Transmitter(object):
    def __init__(self, id=0):
        self.id = id
                                                                           
    def open(self):                                        
        self.axidma_fd = os.open("/dev/axidma", os.O_RDWR | os.O_EXCL)
                                                             
    def close(self):                                                       
        os.close(self.axidma_fd)                                           
                                                                           
    def send_flit_bin(self, flit_bin):
        '''                                                                
        发送flit                                                           
        '''                                                                
        length = len(flit_bin)                                             
        if length > DMA_BUF_MAX:                                           
            print("===<2>=== flit_bin is larger than %dMB" % (DMA_BUF_MAX>>20))
            print("===<2>=== send flit length failed")
            return 0
        print("send flit bytes: %d" % length)
        sent_buf = mmap.mmap(self.axidma_fd,len(flit_bin),mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ)
        sent_buf[:] = flit_bin
        saxidma_transaction            = struct_axidma_transaction()
        saxidma_transaction.wait       = 1
        saxidma_transaction.channel_id = self.id*2
        saxidma_transaction.buf        = bytes2ptr(sent_buf, False)
        saxidma_transaction.buf_len    = len(sent_buf)
        tx_result = fcntl.ioctl(self.axidma_fd, AXIDMA_DMA_WRITE, saxidma_transaction.get())
        return tx_result

    def recv_flit_bin(self, steps = 1, ignore_last = True):
        '''                                                                                                
        接收flit                                                                                           
        '''                                                                                                
        recv = bytearray()
        raxidma_transaction            = struct_axidma_transaction()
        raxidma_transaction.wait       = 1
        raxidma_transaction.channel_id = self.id*2+1
        recv_buf = mmap.mmap(self.axidma_fd,4096,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ)
        raxidma_transaction.buf        = bytes2ptr(recv_buf, False)
        for i in range(steps):
            last = False
            #raxidma_transaction            = struct_axidma_transaction()
            #raxidma_transaction.wait       = 1
            #raxidma_transaction.channel_id = self.id*2+1
            #raxidma_transaction.buf        = bytes2ptr(recv_buf, False)
            raxidma_transaction.buf_len    = len(recv_buf)
            rx_transaction                 = raxidma_transaction.get()
            while not last:
                rx_transaction[12:16] = DMA_RX_LEN
                rx_result = fcntl.ioctl(self.axidma_fd, AXIDMA_DMA_READ, rx_transaction)
                if rx_result == 0:
                    rx_length = struct.unpack('I',rx_transaction[12:16])[0]
                    #print(recv_buf[:rx_length])
                    last_flit = struct.unpack('I',recv_buf[rx_length-4:rx_length])[0]
                    last = last_flit == 0xffffffff
                    if not last or not ignore_last:
                        recv+=recv_buf[:rx_length]
                else:
                    last = True
                    print("[Error] recv result is %d" % rx_result)
        print("recv flit bytes: %d" % len(recv))
        return recv

def recv(trans):
    s1.acquire()
    with open("flitout.bin","wb") as f:
        flits = trans.recv_flit_bin(steps)
        f.write(flits)
    s1.release()
    return flits

def send_and_recv(flit_file):
    with open(flit_file,"rb") as f:
        flit_bin = f.read()
        trans = DMA_Transmitter()
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
        steps = int(sys.argv[2])
    send_and_recv(flit_file)
