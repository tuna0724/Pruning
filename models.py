from importlib.util import module_for_loader
from timm.models import create_model

def mixer_rm_modules(model):
    num_blocks = len(model.blocks)
    
    rm_modules = [(model.blocks[n].mlp_tokens.fc1, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp_tokens.fc2, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp_channels.fc1, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp_channels.fc2, 'weight') for n in range(num_blocks)]
    
    return tuple(rm_modules)

def vit_rm_modules(model):
    num_blocks = len(model.blocks)
    
    rm_modules = [(model.blocks[n].attn.qkv, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].attn.proj, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp.fc1, 'weight') for n in range(num_blocks)]
    rm_modules = rm_modules + [(model.blocks[n].mlp.fc2, 'weight') for n in range(num_blocks)]
    
    return tuple(rm_modules)

def pool_rm_modules(model):
    rm_modules = []

    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[0]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[0]]
    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[2]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[2]]
    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[4]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[4]]
    rm_modules += [(module.mlp.fc1, 'weight') for module in model.network[6]]
    rm_modules += [(module.mlp.fc2, 'weight') for module in model.network[6]]

    
    return tuple(rm_modules)