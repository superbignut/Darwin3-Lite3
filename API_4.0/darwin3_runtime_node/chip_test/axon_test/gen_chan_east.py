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

def west_head(x, y, write = True, target = None, row = True, dir = 0):
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
    sgn_y = dir
    if target is not None:
        if row:
            src_x = abs(x - target)
        else:
            src_y = abs(y - target)
            src_x = 0
    head = (flit_head << 30) + (route_id << 25) + (pkg_class << 22) + (dst_port << 19) + (dst_x << 14) + (sgn_y << 13) + (dst_y << 9) + (src_x << 5) + (src_y << 1)
    fprint(pack('I',head))

def west_core(x, y, write = True, target = None, row = True, ones = True, srams = None, dir = 0):
    if srams is None:
        srams = [
            [0x00014, 0x00001, 4, False],
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
        west_head(x = x, y = y, write = write, target = target, row = row, dir = dir)
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

def scan_read(y_list, row = True, x_list = None, dir = 0):
    if row:
        for y in y_list:
            if x_list is None:
                x_list = range(11,-1,-1) if y > 0 else range(11,0,-1)
            for i in range(len(x_list)-1):
                west_core(x = x_list[i], y = y, write = False, target = x_list[i+1], dir = dir)
    else:
        for y in y_list:
            if x_list is None:
                x_list = range(23,-1,-1) if y > 0 else range(23,0,-1)
            for i in range(len(x_list)-1):
                west_core(y = x_list[i], x = y, write = False, target = x_list[i+1], row = row, dir = dir)

def scan(x = 12):
    # (12, 0) --> (12,23)
    scan_read([x], row = False, x_list = range(0,24), dir = 1)
    # (12,23) --> (13,23)
    scan_read([23], row =  True, x_list = range(x,x+2))
    # (13,23) --> (13, 0)
    scan_read([x+1], row = False, x_list = range(23,-1,-1))
    # (13, 0) --> (14, 0)
    scan_read([  0], row =  True, x_list = range(x+1,x+3))

ones = True
if len(sys.argv) > 1:
    ones = int(sys.argv[1])
start()
# write (12, 0)
west_core(x = 12, y = 0, ones = ones)
scan(12)
scan(14)
scan(16)
scan(18)
scan(20)
scan(22)
stop()
