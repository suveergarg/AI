import socket
import os
import threading
import time

HOST = "127.0.0.1" # localhost
PORT = 11211 # default HTTP port

class Server:
    def __init__(self):
        # Memory data
        self.data = {}

        # Creates a IPv4 TCP socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Socket binds to an IP and port
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # solution for "[Error 89] Address already in use". Use before bind()
        self.socket.bind((HOST, PORT))

        self.socket.listen()
        thread_id = 0
        self.threads = []
        
        try:
            while True:
                # Accept new connections with clients
                conn, addr = self.socket.accept()

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
        while conn:
            request = conn.recv(1024).decode()
             
            if not request:
                break
            
            fields = request.split('\r\n')[0].split(' ')
            byte_count = int(fields[4])

            if fields[0] == 'set':
                data_block = conn.recv(byte_count).decode().split('\r\n')[0]
                self.data[fields[1]] = (data_block, fields[2], fields[3], byte_count, int(time.time()))
                
                if not len(fields) == 6:
                    conn.sendall(b'STORED \r\n')
                
                if int(fields[3]) == -1:
                    del self.data[fields[1]]
            
            if fields[0] == 'get':
                key = fields[1]
                if key in self.data:
                    
                    data_block = self.data[key][0]
                    flags = self.data[key][1]
                    exp_time = int(self.data[key][2])
                    byte_count = self.data[key][3]
                    t = self.data[key][4]

                    if(not exp_time == 0 and int(time.time()) - t > exp_time):
                        del self.data[key]
                        conn.sendall(b'END \r\n')
                    else:    
                        conn.sendall(f'VALUE {key} {byte_count} {flags}\r\n{data_block}\r\nEND\r\n'.encode())
                else:
                    conn.sendall(b'END \r\n')
            
            if fields[0] == 'add':
                data_block = conn.recv(byte_count).decode().split('\r\n')[0]
                if fields[1] in self.data and not len(fields) == 6:
                    conn.sendall(b'NOT_STORED \r\n')
                elif fields[1] not in self.data:
                    self.data[fields[1]] = (data_block, fields[2], fields[3], byte_count, int(time.time()))
                
                    if not len(fields) == 6:
                        conn.sendall(b'STORED \r\n')
                    
                    if int(fields[3]) == -1:
                        del self.data[fields[1]]
            
            if fields[0] == 'replace':
                data_block = conn.recv(byte_count).decode().split('\r\n')[0]
                if fields[1] in self.data:
                    self.data[fields[1]] = (data_block, fields[2], fields[3], byte_count, int(time.time()))
                    if not len(fields) == 6:
                        conn.sendall(b'STORED \r\n')
            
                    if int(fields[3]) == -1:
                        del self.data[fields[1]]
            
            if fields[0] == 'append':
                data_block = conn.recv(byte_count).decode().split('\r\n')[0]
                prev_data_block = self.data[fields[1]][0]
                prev_byte_count = self.data[fields[1]][3]
                prev_time = self.data[fields[1]][-1]
                self.data[fields[1]] = (prev_data_block + data_block , fields[2], fields[3], byte_count + prev_byte_count, prev_time)
                
                if not len(fields) == 6:
                    conn.sendall(b'NOT_STORED \r\n')

            if fields[0] == 'prepend':
                data_block = conn.recv(byte_count).decode().split('\r\n')[0]
                prev_data_block = self.data[fields[1]][0]
                prev_byte_count = self.data[fields[1]][3]
                prev_time = self.data[fields[1]][-1]
                self.data[fields[1]] = (data_block + prev_data_block, fields[2], fields[3], byte_count + prev_byte_count, prev_time)

                if not len(fields) == 6:
                    conn.sendall(b'NOT_STORED \r\n')
        
        conn.close() #Close connection with client
    
server = Server()