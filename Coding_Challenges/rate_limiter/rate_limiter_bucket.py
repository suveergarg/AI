import socket
import os
import threading
import time

HOST = "127.0.0.1" # localhost
PORT = 80 # default HTTP port

class Server:
    def __init__(self):
        # Creates a IPv4 TCP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Socket binds to an IP and port
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
        self.socket.bind((HOST, PORT))

        self.HTTP_header_success = "HTTP/1.1 200 OK\r\n\r\n"
        self.HTTP_header_fail = "HTTP/1.1 400 OK\r\n\r\n Invalid path \r\n\r\n"
        self.HTTP_header_rejected = "HTTP/1.1 429 OK\r\n\r\n Invalid path \r\n\r\n"

        thread_id = 0
        self.threads = []

        self.token_bucket = {}
        self.max_tokens = 10

        self.add_tokens_wait_seconds = 1
        add_token_thread = threading.Thread(target = self.add_tokens).start()

        self.socket.listen()
        
        try:
            while True:
                # Accept new connections with clients
                conn, addr = self.socket.accept()
                conn.settimeout(60) # timeout for clients
                if addr in self.token_bucket and self.token_bucket[addr] == 0:
                    conn.sendall(self.HTTP_header_rejected.encode())
                    conn.close()
                else:
                    
                    if addr not in self.token_bucket:
                        self.token_bucket[addr] = self.max_tokens
                    
                    self.token_bucket[addr] -= 1
                    
                    # Create new threads to service each connection
                    t = threading.Thread(target = self.server_callback, args = (conn, addr, thread_id))
                    t.start()
                    thread_id = (thread_id+1)%1000

        except KeyboardInterrupt:
            print("Stopped by Ctrl+C")
        finally:
            self.socket.close()
            for t in self.threads:
                t.join()

    def server_callback(self, conn, addr, thread_id):
        print(f"Connected by {addr}")

        request = conn.recv(1024).decode()

        print(request)

        header = request.split(" ")
        rate = header[1]

        if(rate == "/unlimited"):
            content = 'Unlimited! \r\n\r\n'
        else:
            content = 'Limited! \r\n\r\n'

        response = self.HTTP_header_success + content
        print('\n Response: ' + response)

        conn.sendall(response.encode())
        conn.close()
    
    def add_tokens(self):
        for ip_addr, tokens in self.token_bucket.items():
            if tokens < self.max_tokens:
                token_bucket[ip_addr] += 1

        threading.Timer(self.add_tokens_wait_seconds, self.add_tokens).start()

server = Server()