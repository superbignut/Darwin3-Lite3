import sys
from struct import pack
def gen_hex(bin_in):
    hex_out = bin_in.replace(".bin","") + ".hex"
    with open(bin_in,"rb") as bin_f:
        hex_f = open(hex_out, "w")
        s=""
        index=0
        for b in bin_f.read():
            s="%02x%s" % (b,s)
            index=index+1
            if(index==4):
                hex_f.write("%s\n" % s)
                s=""
                index=0
        hex_f.close()
    bin_f.close()

if (len(sys.argv) < 2):
    print ("Usage :" + sys.argv[0] + " bin_file")
else:
    gen_hex(sys.argv[1])