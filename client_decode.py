import socket
import sys

clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = socket.gethostname()  
clientsocket.connect((host, 2222))

# do something with the client socket
clientsocket.send(sys.argv[1]+" "+sys.argv[2])
