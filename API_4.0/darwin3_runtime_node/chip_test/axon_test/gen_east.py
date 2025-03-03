import sys
from struct import pack

f = open("/tmp/flitin.bin","wb")

def fprint(bin):
    f.write(bin)
    
def west_body(address, length, size = 2, ones = True, write = True):
    fprint(pack('I',(size << 28) + address+0x8000000))
    fprint(pack('I',length*size))
    if write:
        if ones:
            fprint(pack('I',0xffffffff))
        else:
            fprint(pack('I',0x00000000))
        if size == 6:
            if ones:
                fprint(pack('I',0xffffffff))
            else:
                fprint(pack('I',0x00000000))

def west_head(x, y, write = True, target = None, row = True):
    flit_head = 2
    route_id  = y
    pkg_class = 1 if write else 2
    if x == 23:
        dst_port  = 0
    else:
        dst_port  = 3
    dst_x = 23 - x + 0x10
    dst_y = 0
    src_x = 24 - x
    src_y = 0
    if target is not None:
        if row:
            src_x = target - x
        else:
            src_y = y - target
            src_x = 0
    head = (flit_head << 30) + (route_id << 25) + (pkg_class << 22) + (dst_port << 19) + (dst_x << 14) + (dst_y << 9) + (src_x << 5) + (src_y << 1)
    fprint(pack('I',head))

def west_core(x, y, write = True, target = None, row = True, ones = True, srams = None):
    if srams is None:
        srams = [
            #[0x00014, 0x00001, 4, False],
            [0x00000, 0x00014, 4, False],
            [0x00015, 0x0000b, 4, False],
            [0x00800, 0x00010, 2, ones],
            [0x01000, 0x00080, 2, ones],
            [0x02000, 0x01000, 2, ones],
            [0x04000, 0x02000, 2, ones],
            [0x08000, 0x04000, 4, ones],
            #[0x10000, 0x10000 if big else 0x2000, 6, False],
        ]
    for sram in srams:
        west_head(x = x, y = y, write = write, target = target, row = row)
        west_body(address = sram[0], length = sram[1], size = sram[2], ones = sram[3], write = write)

def start():
    #fprint(pack('I',0xe0010000))
    fprint(pack('I',0xd0000002))

def stop():
    fprint(pack('I',0xc0000001))
    fprint(pack('I',0xc0000000))
    fprint(pack('I',0xd0000000))

def scan_write(y_list, row = True, x_list = None, ones = True, srams = None):
    if row:
         if x_list is None:
             x_list = [11]
         for y in y_list:
             for x in x_list:
                 west_core(x = x, y = y, ones = ones, srams = srams)
    else:
         if x_list is None:
             x_list = [23]
         for y in y_list:
             for x in x_list:
                 west_core(x = y, y = x, ones = ones, srams = srams)

def scan_read(y_list, row = True, x_list = None):
    if row:
        for y in y_list:
            if x_list is None:
                x_list = range(11,-1,-1) if y > 0 else range(11,0,-1)
            for i in range(len(x_list)-1):
                west_core(x = x_list[i], y = y, write = False, target = x_list[i+1])
            west_core(x = x_list[-1], y = y, write = False)
    else:
        for y in y_list:
            if x_list is None:
                x_list = range(23,-1,-1) if y > 0 else range(23,0,-1)
            for i in range(len(x_list)-1):
                west_core(y = x_list[i], x = y, write = False, target = x_list[i+1], row = row)
            west_core(y = x_list[-1], x = y, write = False)

y = 0
ones = True
if len(sys.argv) > 1:
    y = int(sys.argv[1])
if len(sys.argv) > 2:
    ones = int(sys.argv[2])
start()
west_core(x = y, y = 0 if y > 0 else 1, row = False, srams = [[0x00014, 0x00001, 4, False]])
scan_write([y], row = False, ones = ones)
scan_read ([y], row = False)
west_core(x = y, y = 0 if y > 0 else 1, write = False, srams = [[0x00014, 0x00001, 4, False]])
stop()
