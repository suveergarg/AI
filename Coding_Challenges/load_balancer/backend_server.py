import socket
import os
import threading
import time

HOST = "127.0.0.1" # localhost
PORT = 81

class Server:
    def __init__(self):
        # Creates a IPv4 TCP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Socket binds to an IP and port
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
        self.socket.bind((HOST, PORT))

        # Path for HTML documents
        self.root = os.path.join(os.getcwd(), 'www')
        
        # Standard headers
        self.HTTP_header_success = "HTTP/1.1 200 OK\r\n\r\n"
        self.HTTP_header_fail = "HTTP/1.1 400 OK\r\n\r\n Invalid path \r\n\r\n"

        self.socket.listen()
        thread_id = 0
        self.threads = []
        
        try:
            while True:
                # Accept new connections with clients
                conn, addr = self.socket.accept()
                conn.settimeout(60) # timeout for clients
                
                # Create new threads to service each connection
                t = threading.Thread(target = self.run_server, args = (conn, addr, thread_id))
                t.start()
                thread_id = (thread_id+1)%1000
        except KeyboardInterrupt:
            print("Stopped by Ctrl+C")
        finally:
            self.socket.close()
            for t in self.threads:
                t.join()

    def run_server(self, conn, addr, thread_id):
        print(f"Connected by {addr}")
        if conn:
            request = conn.recv(1024).decode()

            if not request:
                conn.close()
                return

        
            print(request)
            print('\n Replied with Hello message \n')            

            response = self.HTTP_header_success + 'Hello From Backend Server' '\n\r\n\r'
            conn.sendall(response.encode())

            conn.close() #Close connection with client

server = Server()