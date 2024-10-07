import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ProbBase(object):
    def __init__(self, model, target_layer, relu, device):
        self.model = model
        self.device = device
        self.relu = relu
        self.model.to(self.device)
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_backward = OrderedDict()
        self.outputs_forward = OrderedDict()
        self.set_hook_func()
    
    def set_hook_func(self):
        raise NotImplementedError
    
    def forward(self, x):
        self.image_size = x.size(-1)
        recon_batch, self.mu, self.logvar = self.model(x)
        return recon_batch, self.mu, self.logvar
    
    def backward(self, statistic):
        self.model.zero_grad()
        self.statistic = statistic.to(self.device)
        self.statistic.backward(retain_graph=True)
    
    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            #print(f'key:{key}, value:{value}')
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError(f'Invalid layer name: {target_layer}')


class GradCAM(ProbBase):
    def set_hook_func(self):
        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()
            
        #print(f'outputs_backward:{self.outputs_backward.values()}\n')
        
        def func_f(module, input, f_output):
            self.outputs_forward[id(module)] = f_output
            
        #print(f'outputs_forward:{self.outputs_forward.values()}\n')
        
        for module in self.model.named_modules():
            if module[0] == self.target_layer:
                module[1].register_backward_hook(func_b) # saves output of backward pass
                module[1].register_forward_hook(func_f)  # saves output of forward pass
    
    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.item()
    
    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads.squeeze())
        self.map_size = self.grads.size()[2:]
        self.alpha = nn.AvgPool2d(self.map_size)(self.grads)
    
    def generate(self):
        # shape:torch tensor [128, 64, 16, 16]
        self.grads = self.get_conv_outputs(self.outputs_backward, self.target_layer)
        #print(f'\nGetting gradient in generate:{self.grads.shape}')
        
        # compute weithts based on the gradient:a_{k}^{c} = GAP(dy^{c}/d(A_{ij}^{k}))
        # get alpha=weights: torch_tensor:([128, 64, 1, 1]),torch.float32
        self.compute_gradient_weights()
        #print(f'\nComputing coefficients alpha using GAP:{self.alpha.shape},{self.alpha.dtype}')
        
        # get activation A_k
        # shape:torch tensor [128, 64, 16, 16]
        self.activation = self.get_conv_outputs(self.outputs_forward, self.target_layer)
        #print(f'\nShape and type activateions A_k:{self.activation.shape},{self.activation.dtype}')
        
        self.activation = self.activation[None, :, :, :, :]
        self.alpha = self.alpha[:, None, :, :, :]
        
        # gcam: torch.Size([1, 128, 1, 16, 16]), torch.float32
        # in original paper we have ReLU, but we dot have it here
        # L_{gcam}^{c} = ReLU(sum_{a_{k}A_{k}})
        gcam = F.conv3d(self.activation, (self.alpha.cuda()), padding=0, groups=len(self.alpha))
        #print(f'\nShape of heatmap after mul alpha_kxA_k:{gcam.shape},{gcam.dtype}')
        
        # gcam size: [128, 1, 16, 16]
        gcam = gcam.squeeze(dim=0)
        #print(f'\ngcam size:{gcam.size()}')
        # Upsample gcam to the size of image: [128, 1, 64, 64]
        gcam = F.interpolate(gcam, (self.image_size, self.image_size), mode="bilinear")
        #print(f'\nsize of heatmap after upsampling:{gcam.size()}')
        # gcam size: [128, 1, 64, 64]
        # applying relue to produce features that positively impact test statistic
        if self.relu:
            gcam = F.relu(gcam)
            print('ReLU was applied!')
        else:
            gcam = torch.abs(gcam)
        #print(f'\nsize of heatmap after abs value:{gcam.size()}')
        return gcam


