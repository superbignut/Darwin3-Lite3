import time, sys

def run_parser(recv_flitout='recv_flitout.txt'):
    print('========= parser recv flit : %s' % recv_flitout)
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
                x = ((eval("0x"+items[5:7])>>1)&0xf) - 1
                is_dedr = 0
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
    print('========= parser recv flit elapsed : %.3f ms' % ((end_time - start_time)/1000000))

if __name__ == '__main__':
    recv_flitout = 'recv_flitout.txt'
    if len(sys.argv) > 1:
        recv_flitout = sys.argv[1]
        if recv_flitout == '-h' or recv_flitout == '--help':
            print("Usage:\n      python %s flits_file" % sys.argv[0])
            sys.exit()
    run_parser(recv_flitout)