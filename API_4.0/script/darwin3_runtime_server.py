#!python
import socket,sys,struct,os,time,threading,subprocess,logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s]<%(name)s> --> %(message)s")
import mmap
import fcntl
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi,urllib
from functools import partial

darwin3_runtime_node_dir = sys.prefix + "/darwin3_runtime_node"

SEMAPHORE1 = threading.Semaphore(1)
IP = ""
CONTROL_PORT=6001
DATA_PORT=6000

MSG_DEPLOY=0x7000
MSG_INPUT=0x7001
EMPTY_BYTES=b''
PACKAGE_HEAD_LEN=20

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
        self._has_opened=False
        if self.id >= 2:
            self.id = self.id + 1
        self.tx_dma_fd=-1
        self.rx_dma_fd=-1
    
    def has_open(self):
        return self._has_opend
                                                                           
    def open(self):
        if self._has_opened:
            return
        else:
            if self.id == 0:                                           
                self.tx_dma_fd = os.open("/dev/dma_proxy_tx", os.O_RDWR)           
                self.rx_dma_fd = os.open("/dev/dma_proxy_rx", os.O_RDWR)
            else:
                self.tx_dma_fd = os.open("/dev/dma_proxy_tx%d" % self.id, os.O_RDWR)           
                self.rx_dma_fd = os.open("/dev/dma_proxy_rx%d" % self.id, os.O_RDWR)  
            self._has_opened=True         
                                                                           
    def close(self):
        if self._has_opened:                                                     
            os.close(self.tx_dma_fd)                                           
            os.close(self.rx_dma_fd) 
            self.tx_dma_fd=-1
            self.rx_dma_fd=-1
            self._has_opened=False      
            
    def __str__(self):
        return f"DMA_Transmitter(DMA_id={self.id}, tx_dma_fd={self.tx_dma_fd}, rx_dma_fd={self.rx_dma_fd})"                                    
                                                                           
    def send_flit_bin(self, flit_bin):
        '''                                                                
        发送flit                                                           
        '''                                                                
        length = len(flit_bin)                                             
        if length > DMA_BUF_MAX:                                           
            logging.error("===<2>=== %s is larger than %dMB" % (flit_bin,DMA_BUF_MAX>>20))
            logging.error("===<2>=== send flit length failed")                                 
            return 0                                                                   
        logging.info("flit length: %d" % length)                                 
        with mmap.mmap(self.tx_dma_fd,DMA_MAP_SIZE,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ) as mm:
            mm[DMA_LENGTH:DMA_LENGTH+4]=struct.pack('I',length)                                            
            mm[:length]=flit_bin                                                                           
            fcntl.ioctl(self.tx_dma_fd, XFER, DMA_BINDEX)                                                  
            tx_result = struct.unpack('I',mm[DMA_RES:DMA_RES+4])[0]                                        
            if tx_result != 0:                                                                             
                return 0                                                                                   
        return 1 

    def recv_flit_bin(self):
        '''                                                                                                
        接收flit                                                                                           
        '''                                                                                                
        recv = bytearray()                                                                                       
        with mmap.mmap(self.rx_dma_fd,DMA_MAP_SIZE,mmap.MAP_SHARED,mmap.PROT_WRITE | mmap.PROT_READ) as mm:
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
                        rx_length-=4
                    recv+=mm[:rx_length]
                else:
                    last = True
            else:                                                                                          
                last = True                                                                                
        return recv 
        
    @staticmethod
    def read_and_send_via_tcp(dma,tcp_client,show_data):
        SEMAPHORE1.acquire()
        flits = dma.recv_flit_bin()
        tcp_client.sendall(flits)
        SEMAPHORE1.release()
        if show_data:
            logging.getLogger(f'DMA_Transmitter_{dma.id}').info(flits)
        return

DIRECTION_TRANSMITTERS=[DMA_Transmitter(_i) for _i in range(4)]
WEST_DMA=DIRECTION_TRANSMITTERS[0]
EAST_DMA=DIRECTION_TRANSMITTERS[1]
NORTH_DMA=DIRECTION_TRANSMITTERS[2]
SOUTH_DMA=DIRECTION_TRANSMITTERS[3]

def chip_reset():
    for _dma in DIRECTION_TRANSMITTERS: _dma.close()
    os.system("bash " + darwin3_runtime_node_dir + "/script/restart_dma")
    os.system("bash " + darwin3_runtime_node_dir + "/script/restart_dma")
    for _dma in DIRECTION_TRANSMITTERS: _dma.open()

def set_fraquency(freq):
    for _dma in DIRECTION_TRANSMITTERS: _dma.close()
    subprocess.run([darwin3_runtime_node_dir + "/script/flits_sender", darwin3_runtime_node_dir + f"/chip_test/reset_clock/pll_clock_{freq}M.bin"],check=True)  # 如果命令失败则引发异常
    for _dma in DIRECTION_TRANSMITTERS: _dma.open()

def start_tcp_server(ip, port):
    global IP
    # create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)
 
    data_tcp_logger=logging.getLogger('data_TCP_server')
    # bind port
    data_tcp_logger.info("Starting listen on ip %s, port %s" % server_address)
    sock.bind(server_address)
 
    # start listening, allow only one connection
    try:
        sock.listen(1)
    except Exception as e:
        data_tcp_logger.critical(f"fail to listen on port: {e}")
        sys.exit(1)
    receive_count = 0

    thread_pool=ThreadPoolExecutor(4)
    while True:

        data_tcp_logger.info("waiting for connection")
        try:
            client, addr = sock.accept()
            _stime = time.time()
        except:
            print("Kill by user!")
            sock.close() 
            sys.exit(1)
        data_tcp_logger.info(f"incoming connection: {addr}")

        try:
            _in_data = client.recv(RECV_SIZE)
        except socket.error:
            data_tcp_logger.error("\nlength error\n")
            for _dma in DIRECTION_TRANSMITTERS: _dma.close()
            break
            
        for _dma in DIRECTION_TRANSMITTERS: _dma.open()

        if len(_in_data) < PACKAGE_HEAD_LEN:
            client.close()
            data_tcp_logger.error(f'Erroneous msg: {_in_data}')
            break
        # type, w_len,e_len,n_len,s_len
        _dtype,_w_data_len,_e_data_len,_n_data_len,_s_data_len=struct.unpack('IIIII',_in_data[:PACKAGE_HEAD_LEN])
     
        if _dtype==MSG_DEPLOY:
            data_tcp_logger.info('receive deployment data')
        else:
            data_tcp_logger.info('receive spikes')
        logging.debug(f"data_type={_dtype:08x}, _w_len={_w_data_len}, _e_len={_e_data_len}, _n_len={_n_data_len}, _s_len={_s_data_len}")


        _spike_start_time=time.time()
        if _w_data_len>0:
            data_tcp_logger.debug('send west')
            WEST_DMA.send_flit_bin(_in_data[
                PACKAGE_HEAD_LEN:PACKAGE_HEAD_LEN+_w_data_len
            ])
        if _e_data_len>0:
            data_tcp_logger.debug('send east')
            EAST_DMA.send_flit_bin(_in_data[
                PACKAGE_HEAD_LEN+_w_data_len:PACKAGE_HEAD_LEN+_w_data_len+_e_data_len
            ])
        if _n_data_len>0:
            data_tcp_logger.debug('send north')
            NORTH_DMA.send_flit_bin(_in_data[
                PACKAGE_HEAD_LEN+_w_data_len+_e_data_len:PACKAGE_HEAD_LEN+_w_data_len+_e_data_len+_n_data_len
            ])
        if _s_data_len>0:
            data_tcp_logger.debug('send south')
            SOUTH_DMA.send_flit_bin(_in_data[
                PACKAGE_HEAD_LEN+_w_data_len+_e_data_len+_n_data_len:
            ])

        _w_output=EMPTY_BYTES
        _e_output=EMPTY_BYTES
        _n_output=EMPTY_BYTES
        _s_output=EMPTY_BYTES
        if _w_data_len>0:
            data_tcp_logger.debug('read west')
            _w_output=WEST_DMA.recv_flit_bin()
        if _e_data_len>0:
            data_tcp_logger.debug('read east')
            _e_output=EAST_DMA.recv_flit_bin()
        if _n_data_len>0:
            data_tcp_logger.debug('read north')
            _n_output=NORTH_DMA.recv_flit_bin()
        if _s_data_len>0:
            data_tcp_logger.debug('read south')
            _s_output=SOUTH_DMA.recv_flit_bin()
        
        Control_Server.LAST_RUNNING_TIME=time.time()-_spike_start_time
        
        _output_data=EMPTY_BYTES.join([
            struct.pack('IIII',len(_w_output),len(_e_output),len(_n_output),len(_s_output)),
            _w_output,_e_output,_n_output,_s_output])
        print(_e_output)
        
        data_tcp_logger.debug(_output_data)
        client.sendall(_output_data)

        Control_Server.LAST_TCP_TIME=time.time()-_stime


from functools import wraps
GET_METHODS={}


def GET(url,content_type='text/plain'):
    def out_wrapper(func):
        @wraps(func)
        def _inner(self,*args,**kwds):
            content=func(self,*args,**kwds)
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.end_headers()
            if content is None: 
                content=''
            self.wfile.write(content.encode())
        GET_METHODS[url]=_inner
        return _inner
    return out_wrapper
        
class Control_Server(BaseHTTPRequestHandler):
    
    LAST_RUNNING_TIME=-1.
    LAST_TCP_TIME=-1.

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        logging.debug(parsed_path.query)
        post_vars = urllib.parse.parse_qs(parsed_path.query)
        GET_METHODS[parsed_path.path](self,**post_vars)
    
    @GET('/',content_type='text/html')
    def get_main_page(self):
        return f"""
        <html>
        <head><title>下位机</title></head>
        <br>generating_spike_time_span: {Control_Server.LAST_RUNNING_TIME}</br>
        <br>tcp_total_time_span: {Control_Server.LAST_TCP_TIME}</br>
        <body>
        </body>
        </html>
        """

    @GET('/last_running_time')
    def get_running_time(self):
        return str(Control_Server.LAST_RUNNING_TIME)

    @GET('/last_tcp_time')
    def get_running_time(self):
        return str(Control_Server.LAST_TCP_TIME)
    
    @GET('/chip_reset')
    def get_reset_chip(self):
        chip_reset()

    @GET('/set_frequency')
    def get_set_fraquency(self,freq):
        logging.info(f"set freq={freq}M")
        set_fraquency(freq[0])

if __name__=='__main__':
    
    control_server= HTTPServer((IP,CONTROL_PORT),Control_Server)
    threading.Thread(target=control_server.serve_forever,daemon=True).start()
    chip_reset()
    while(True):
        try:
            server_thread1 = threading.Thread(target=start_tcp_server, args=(IP,DATA_PORT),daemon=True)
            server_thread1.start()
        except socket.error:
            logging.getLogger('data_tcp_server').info(f"reconnecting")

        server_thread1.join()
        time.sleep(1)
