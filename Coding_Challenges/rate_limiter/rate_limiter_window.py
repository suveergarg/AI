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

        self.time_window = {}
        self.request_threshold = 60
        self.request_counter = 0

        self.socket.listen()
        
        try:
            while True:
                # Accept new connections with clients
                conn, addr = self.socket.accept()
                conn.settimeout(60) # timeout for clients
                
                current_window = floor(time.time())

                # Add current window into the dictionary
                if current_window not in self.time_window:
                    self.time_window[current_window] = 0
                    to_remove = current_window - 60
                    
                    # Removes window 60s older
                    if to_remove in self.time_window:
                        self.request_counter -= self.time_window[to_remove]
                        del self.request_counter[to_remove]

                 self.time_window[current_window] += 1

                if self.request_counter > self.request_threshold:
                    conn.sendall(self.HTTP_header_rejected.encode())
                    conn.close()
                
                else:
                    self.time_window[current_window] += 1
                    
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
    
server = Server()