import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

SCORE = None

class Pruner(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
    
    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        print('number of parameters:', tensor_size)
        print('number of parameters to prune:', nparams_toprune)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        global SCORE
        if nparams_toprune != 0:
            topk = torch.topk(SCORE, k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0

        return mask

def random(rm_modules):
    scores = [torch.rand_like(module.weight).view(-1) for module, _ in rm_modules]
    scores = torch.cat(scores)

    return scores

def magnitude(rm_modules):
    scores = [module.weight.abs().view(-1) for module, _ in rm_modules]
    scores = torch.cat(scores)

    return scores

def snip(model, rm_modules, dataloader, device):
    device = [i for i in range(device)]
    model = nn.DataParallel(model, device_ids=device).to('cuda:0')

    rm_weights = [module.weight for module, _ in rm_modules]

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(0), targets.cuda(0)
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)
    
    model.zero_grad()
    return score

def snip_magnitude(model, rm_modules, dataloader, device, alpha):
    device = [i for i in range(device)]
    model = nn.DataParallel(model, device_ids=device).to('cuda:0')

    rm_weights = [module.weight for module, _ in rm_modules]

    with torch.no_grad():
        grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(0), targets.cuda(0)
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        grads = list(torch.autograd.grad(loss, rm_weights))
        break

    with torch.no_grad():
        score = [(weight.cpu() * grad.cpu()).view(-1).abs() + alpha * (weight.view(-1).cpu() * weight.view(-1).cpu()) for weight, grad in zip(rm_weights, grads)]
        score = torch.cat(score)
    
    model.zero_grad()
    return score

def grasp(model, rm_modules, dataloader, device):
    device = [i for i in range(device)]
    model = nn.DataParallel(model, device_ids=device).to('cuda:0')

    rm_weights = [module.weight for module, _ in rm_modules]
    
    with torch.no_grad():
        stopped_grads = [torch.zeros_like(w) for w in rm_weights]
    
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.cuda(0), targets.cuda(0)
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        grads = list(torch.autograd.grad(loss, rm_weights, create_graph=True))
        break
    
    V = 0.0
    for g in grads:
        V += g.reshape(-1).clone().detach() @ g.reshape(-1)
    V.backward()
            
    with torch.no_grad():
        grads = torch.cat([w.grad.view(-1).detach().cpu() for w in rm_weights])
        Weights = torch.cat([w.view(-1).detach().cpu() for w in rm_weights])
        score = (Weights * grads).clone().detach()
    
    model.zero_grad()
    return score
