import socket
import threading
import traceback
import sys
def handle_client_thread(client_socket):
    # 处理线程
    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            command, args1, args2 = data.decode('utf-8').split()  # 假设数据格式为 "COMMAND arg1 arg2"
            print(f"Received command: {command}, args: {args1}, {args2}")
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info() # 返回报错信息
        # Traceback objects represent the stack trace of an exception. A traceback object is implicitly created
        # when an exception occurs, and may also be explicitly created by calling types.TracebackType.
        print(traceback.format_exception(
            exc_type,
            exc_value,
            exc_traceback
        ))
    finally:
        client_socket.close()

            

def start_server(host='localhost', port=12345):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5) # 等待数， 连接数
    print(f"Server listening on {host}:{port}...")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        client_handler = threading.Thread(target=handle_client_thread, args=(client_socket,))
        client_handler.start()

if __name__ == "__main__":
    start_server()
