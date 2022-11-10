# Pruning

## Requirements
```
pip3 install -r requirements.txt
```
The code was tested with Python 3.8.10


## Models
The following are models that can be used.
- vit_base_patch16_224
- vit_large_patch16_224
- mixer_b16_224
- mixer_l16_224
- poolformer_m36
- poolformer_m48

## How to run?
```
# CIFAR-10, Mixer-B/16, Sparsity = 95%
$ python3 main.py --model mixer_b16_224 --pretrain --dataset cifar10 --pruning --sparsity 95 --method snip_magnitude --alpha 0.001

# SNIP, CIFAR-100, ViT-B/16, Sparsity = 95%
$ python3 main.py --model vit_base_patch16_224 --pretrain --dataset cifar100 --pruning --sparsity 95 --method snip_magnitude --alpha 0.001
```
