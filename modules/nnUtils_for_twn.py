import torch.nn as nn
import torch
from torch.autograd import Function,Variable
import pdb

def compute_threshold(tensor, linear=False):
    s = tensor.size()
    n = tensor[0].nelement()

    if not linear:
        tmp = tensor.norm(1, 3, keepdim=True) \
            .sum(2, keepdim=True).sum(1, keepdim=True)
    else:
        tmp = tensor.norm(1, 1, keepdim=True)

    delta = tmp.div(n).mul(0.7)
    return delta

def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

def compute_alpha(x,linear=False):
    threshold = compute_threshold(x,linear)
    alpha1_temp1 = where(torch.ge(x,threshold), x, torch.zeros_like(x))
    alpha1_temp2 = where(torch.le(x,-threshold), x, torch.zeros_like(x))
    alpha_array = torch.add(alpha1_temp1,alpha1_temp2)
    alpha_array_abs = torch.abs(alpha_array)
    alpha_array_abs1 = where(alpha_array_abs>0,torch.ones_like(alpha_array_abs), torch.zeros_like(alpha_array_abs))
    if not linear:
        #alpha_sum = torch.sum(alpha_array_abs)
        alpha_sum = alpha_array_abs.sum(3, keepdim=True) \
            .sum(2, keepdim=True).sum(1, keepdim=True)
        #n = torch.sum(alpha_array_abs1)
        n = alpha_array_abs1.sum(3, keepdim=True) \
            .sum(2, keepdim=True).sum(1, keepdim=True)
    else:
        #alpha_sum = torch.sum(alpha_array_abs)
        alpha_sum = alpha_array_abs.sum(1, keepdim=True)
        #n = torch.sum(alpha_array_abs1)
        n = alpha_array_abs1.sum(1, keepdim=True)
    alpha = torch.div(alpha_sum,n)
    return alpha


class tenarize_conv2d(Function):

    def __init__(self, is_activation):
        super(tenarize_conv2d, self).__init__()
        self.is_activation = is_activation

    def forward(self, inputs):

        tensor = inputs.clone()
        self.forward_tensor = tensor

        delta = compute_threshold(tensor)
        alpha = compute_alpha(tensor)

        tensor[tensor.ge(delta)] = 1.
        tensor[tensor.le(-1. * delta)] = -1.
        tensor[tensor.abs().ne(1)] = 0.

        #alpha = tmp.div(tensor.abs().norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True))

        tensor = tensor.mul(alpha)
        #inputs.data = tensor
        self.alpha=alpha
        #pdb.set_trace()
        return tensor

    def backward(self, grad_output):
        inputs = self.forward_tensor
        grad_input = grad_output.clone()
        #print(grad_input.size())
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        return grad_input


class tenarize_linear(Function):

    def __init__(self, is_activation):
        super(tenarize_linear, self).__init__()
        self.is_activation = is_activation

    def forward(self, inputs):
        tensor = inputs.clone()
        self.forward_tensor = tensor
        delta = compute_threshold(tensor,linear=True)
        alpha = compute_alpha(tensor,linear=True)

        tensor[tensor.ge(delta)] = 1.
        tensor[tensor.le(-1. * delta)] = -1.
        tensor[tensor.abs().ne(1)] = 0.

        #alpha = tmp.div(tensor.abs().norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True))

        tensor = tensor.mul(alpha)
        #inputs.data = tensor
        self.alpha=alpha
        #pdb.set_trace()
        return tensor

    def backward(self, grad_output):
        inputs= self.forward_tensor
        grad_input = grad_output.clone()
        #if not self.is_activation:
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        return grad_input


class BinLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinLinear, self).__init__(*kargs, **kwargs)
        c = float(self.weight.data[0].nelement())
        self.weight.data = self.weight.data.normal_(0, 1.0 / c)

    def forward(self, input):
        # if input.size(1) != 784:
        #    input.data=Binarize(input.data)
        #self.ori_weight = self.weight.clone()
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        bin_weight = tenarize_linear(False)
        weight_bin = bin_weight(self.weight)

        out = nn.functional.linear(input, weight_bin)

        '''if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            self.bias.data = torch.sign(self.bias.data).mul(alpha.view(alpha.size(0)))
            out += self.bias.view(1, -1).expand_as(out)'''

        return out


class BinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)
        c = float(self.weight.data[0].nelement())
        self.weight.data = self.weight.data.normal_(0, 1.0 / c)
        #self.delta = compute_threshold(self.weight,False).cuda()
        #self.delta = nn.Parameter(delta)


    def forward(self, input):
        # if input.size(1) != 3:
        #    input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        if not hasattr(self, 'delta'):
            delta = compute_threshold(self.weight,False)
            self.delta = nn.Parameter(delta)#'''

        #bin_weight = tenarize_conv2d(False)
        # weight_bin = bin_weight(self.weight)

        tensor =self.weight.data.clone()
        alpha = compute_alpha(tensor)

        tensor[tensor.ge(self.delta)] = 1.
        tensor[tensor.le(-1. * self.delta)] = -1.
        tensor[tensor.abs().ne(1.)] = 0.

        #print(self.delta[0])

        bin_w = tensor.mul(alpha)

        out = nn.functional.conv2d(input, bin_w, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        '''if not self.bias is None:
            alpha = bin_weight.alpha
            self.bias.org=self.bias.data.clone()
            self.bias.data = torch.sign(self.bias.data).mul(alpha.view(alpha.size(0)))
            #pdb.set_trace()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)'''

        return out