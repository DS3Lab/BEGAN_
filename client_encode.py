import socket
import sys

clientsocket = socket.socket()
host = socket.gethostname()  
clientsocket.connect((host, 2222))

# send soemthing
clientsocket.send(sys.argv[1])
