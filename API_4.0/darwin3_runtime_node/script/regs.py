import os,struct
import sys

addr=0

if len(sys.argv) > 1:
    addr=int(sys.argv[1])

reg_fd = os.open("/dev/axi-regs", os.O_RDWR)
if len(sys.argv) > 2:
    new_value = int(sys.argv[2], 16)
    data = struct.pack("I", new_value)
    os.lseek(reg_fd, addr, os.SEEK_SET)
    os.write(reg_fd, data)
# 读取寄存器的值
os.lseek(reg_fd, addr, os.SEEK_SET)
data = os.read(reg_fd, 4)
value = struct.unpack("I", data)[0]
print("[0x%02x] = 0x%08x" % (addr, value))
os.close(reg_fd)

