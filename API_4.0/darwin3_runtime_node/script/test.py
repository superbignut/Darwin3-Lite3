import mmap
import os
import struct
import fcntl
import sys
import time
import threading

prefix = ""
if len(sys.argv) > 1:
    prefix = sys.argv[1]
if prefix != "":
    prefix += "_"

FINISH_XFER = ord('a') + (ord('a') << 8) + (4 << 16) + (1 << 30)
START_XFER  = ord('b') + (ord('a') << 8) + (4 << 16) + (1 << 30)
XFER        = ord('c') + (ord('a') << 8) + (4 << 16) + (2 << 30)

tx_dma_fd = os.open("/dev/dma_proxy_tx", os.O_RDWR)

with open(prefix+'flitin.bin','rb') as f:
    bin=f.read()
    l=len(bin)

with mmap.mmap(tx_dma_fd,4*1024*1024+8,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ) as mm:
    mm[4*1024*1024+4:4*1024*1024+8]=struct.pack('I',l)
    mm[:l]=bin
    fcntl.ioctl(tx_dma_fd, XFER, b'\x00\x00\x00\x00')
    tx_result = struct.unpack('I',mm[4*1024*1024:4*1024*1024+4])[0]
    print("tx result is %d" % tx_result)
    print("tx length is %d" % l)

def recv(l):
    rx_dma_fd = os.open("/dev/dma_proxy_rx", os.O_RDWR)
    with mmap.mmap(rx_dma_fd,4*1024*1024+8,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ) as mm:
        mm[4*1024*1024+4:4*1024*1024+8]=struct.pack('I',l)
        fcntl.ioctl(rx_dma_fd, XFER, b'\x00\x00\x00\x00')
        rx_result = struct.unpack('I',mm[4*1024*1024:4*1024*1024+4])[0]
        print("rx result is %d" % rx_result)
        if rx_result == 0:
            rx_length = struct.unpack('I',mm[4*1024*1024+4:4*1024*1024+8])[0]
            print("rx length is %d" % rx_length)

thread = threading.Thread(target=recv, args=(l,))
thread.start()
