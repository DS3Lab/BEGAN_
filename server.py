import socket

import sys
import numpy as np
import json
import scipy.misc
import tensorflow as tf
from trainer import Trainer
from config import get_config
from data_loader import get_loader
from utils import prepare_dirs_and_logger


# Server: initialize training, build model, build test model; listen
# if only 1 argument, do the encoding and sends back the z vector; if 2 arguments, do decoding and save the image
# Client: if needs encoding, sends the image; if needs decoding, sends the json file and desired image file name

#------------------------first, build training models------------------------------------------------------------
config, unparsed = get_config()
prepare_dirs_and_logger(config)
rng = np.random.RandomState(config.random_seed)
tf.set_random_seed(config.random_seed)
data_loader = get_loader(config.batch_size, config.input_scale_size,
        config.data_format, config.split)
trainer = Trainer(config, data_loader)


s = socket.socket()
host = socket.gethostname()  
s.bind((host, 2222)) 
s.listen(5)  # max 5


while True:  # always working
	client_socket, addr = s.accept()     # Establish connection with client.
	buf = client_socket.recv(512*4)
	buf = buf.split() 
#	print("The request received from the client is: " + buf[0])
        print("Only support 64 * 64 images for now\n")
	if len(buf) == 1:
   	# encoder
   		trainer.encoder(buf[0])  # will print out the json z_vector and save it at the same time
	else:
		trainer.decoder(buf[0], buf[1]) # will save the generated file

 



