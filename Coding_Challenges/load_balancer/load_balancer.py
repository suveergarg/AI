import socket
import os
import threading
import time

HOST = "127.0.0.1" # localhost
PORT = 80 # default HTTP port
BACKENDPORT = 81

class Server:
    def __init__(self):
        # Creates a IPv4 TCP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Socket binds to an IP and port
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
        self.socket.bind((HOST, PORT))

        thread_id = 0
        self.threads = []

        # Creating a backend port
        self.backend_ports = [8080, 8081, 8082]
        self.backend_port_health = {
            8080 : False, # Meaning Down
            8081 : False,
            8082 : False 
        }

        self.health_check_wait_seconds = 3
        health_check_thread = threading.Thread(target = self.backend_server_health_check).start()

        self.socket.listen()
        
        try:
            while True:
                # Accept new connections with clients
                conn, addr = self.socket.accept()
                conn.settimeout(60) # timeout for clients

                # Queue for backend ports
                port = self.backend_ports.pop(0)
                self.backend_ports.append(port)
                while not self.backend_port_health[port]:
                    port = self.backend_ports.pop(0)
                    self.backend_ports.append(port)
                    print("Finding the next healthy server")
                
                # Create new threads to service each connection
                t = threading.Thread(target = self.send_to_server, args = (conn, addr, port, thread_id))
                t.start()
                thread_id = (thread_id+1)%1000

        except KeyboardInterrupt:
            print("Stopped by Ctrl+C")
        finally:
            self.socket.close()
            for t in self.threads:
                t.join()

    def send_to_server(self, conn, addr, port, thread_id):
        print(f"Connected by {addr}")

        request = conn.recv(1024)

        print(request.decode())

        backend_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        backend_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
        backend_socket.connect((HOST, port))
        backend_socket.sendall(request)

        response = ""
        while True: 
            response_part = backend_socket.recv(1024)
            if not response_part:
                break

            response += response_part.decode()


        print('\n Respose from Server: ' + response)

        conn.sendall(response.encode())
        conn.close()
    
    def backend_server_health_check(self):
        request = b'GET / HTTP/1.1\r\nHost: localhost\r\nUser-Agent: HealthChecker\r\nAccept: */*\r\n\r\n'
        for port, health in self.backend_port_health.items():
            backend_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            backend_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
            
            try:
                backend_socket.connect((HOST, port))
                backend_socket.sendall(request)
                response = backend_socket.recv(1024).decode()
            except:
                self.backend_port_health[port] = False
                continue
            
            if(response.split(" ")[1] == "200"):
                self.backend_port_health[port] = True
            else:
                self.backend_port_health[port] = False
        
        threading.Timer(self.health_check_wait_seconds, self.backend_server_health_check).start()

server = Server()