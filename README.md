# BEGAN_

This is an autoencoder for better manipulation of the embedded 'z-vector' after encoding.

The model is trained with the original BEGAN code[1] after 466017 steps.

The architecture is exactly the same as that in [2] (the discriminator part).

## Usage

### Preparation

To place the model in the right path:
```
git clone https://github.com/DS3Lab/BEGAN_.git
cd BEGAN_
mv /mnt/ds3lab/litian/logs .  
```

### Start to use

Start the server:

```
CUDA_VISIBLE_DEVICES=X python server.py
```

To encode (now in the config, the file size is 64 * 64):

```
CUDA_VISIBLE_DEVICES=X python client_encode.py file_path/file_name
```
The server will print the z-vector and save it as z.json at the same time.

To decode:
```
CUDA_VISIBLE_DEVICES=X python client_decode.py z.json generated_file_name
```
The server will save the generated image 'generated_file_name' in the current directory.


## References
[1] https://github.com/carpedm20/BEGAN-tensorflow

[2] BEGAN: Boundary Equilibrium Generative Adversarial Networks.
