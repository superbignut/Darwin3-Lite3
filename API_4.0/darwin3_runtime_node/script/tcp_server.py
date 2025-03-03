import socket
import sys
import struct
import os
import time
import mmap
import fcntl
import threading

active_ip = ""
RECV_SIZE = 4194304

def start_tcp_server(ip, port, id):
    global active_ip
    # create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)
 
    # bind port
    print("==Starting listen on ip %s, port %s" % server_address)
    sock.bind(server_address)
 
    # start listening, allow only one connection
    try:
        sock.listen(1)
    except socket.error:
        print("fail to listen on port %s" % e)
        sys.exit(1)
    receive_count = 0

    while True:
        recv_len  = 0
        length    = 0

        while True:
            print("\nwaiting for connection")
            try:
                client, addr = sock.accept()
                stime = time.time_ns()
                #trans.open()
            except:
                print("Kill by user!")
                sock.close() 
                sys.exit(1)
            #print("having a connection")
            print("addr is ", addr)
            break

        while True:
            try:
                msg = client.recv(RECV_SIZE)
            except socket.error:
                print("\nlength error\n")
                trans.close()
                length = 0
                break

            recv_len += len(msg)
            
            if recv_len >= RECV_SIZE:
                client.close()
                etime = time.time_ns()                                              
                print("speed is %.3f Mbps" % (8*recv_len*1000000000/(etime-stime)/2**20))
                recv_len = 0
                break
                
        if (length == 0):
            active_ip = ""
 
    print("\n==Finish, close connect")
    sock.close() 

if __name__=='__main__':
    while(True):
        active_ip = ""
        id = 0
        if len(sys.argv)>1:
            id = int(sys.argv[1])
        try:
            start_tcp_server('0.0.0.0',7+id,id)
        except socket.error:
            print("\nreconnect\n")
        time.sleep(1)
