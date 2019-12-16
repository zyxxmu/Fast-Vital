import torch.nn as nn
import torch
from torch.autograd import Function,Variable
import pdb


class binarize_conv2d(Function):

    def __init__(self, is_activation):
        super(binarize_conv2d, self).__init__()
        self.is_activation = is_activation

    def forward(self, inputs):

        tensor = inputs.clone()
        self.forward_tensor = tensor
        tensor = tensor.sign()

        return tensor

    def backward(self, grad_output):
        inputs = self.forward_tensor
        grad_input = grad_output.clone()
        #pdb.set_trace()
        #print(grad_input.size())
        #if not self.is_activation:
        grad_input[inputs.ge(1)] = 0
        grad_input[inputs.le(-1)] = 0
        return grad_input


class binarize_linear(Function):

    def __init__(self, is_activation):
        super(binarize_linear, self).__init__()
        self.is_activation = is_activation

    def forward(self, inputs):

        tensor = inputs.clone()
        self.forward_tensor = tensor
        tensor = tensor.sign()
        return tensor

    def backward(self, grad_output):
        inputs= self.forward_tensor
        grad_input = grad_output.clone()
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

        bin_weight = binarize_linear(False)
        weight_bin = bin_weight(self.weight)
        input_bin = binarize_linear(True)(input)

        #pdb.set_trace()

        out = nn.functional.linear(input_bin, weight_bin)

        '''if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            self.bias.data = torch.sign(self.bias.data)
            out += self.bias.view(1, -1).expand_as(out)'''

        return out


class BinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)
        c = float(self.weight.data[0].nelement())
        self.weight.data = self.weight.data.normal_(0, 1.0 / c)

    def forward(self, input):
        # if input.size(1) != 3:
        #    input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        bin_weight = binarize_conv2d(False)
        weight_bin = bin_weight(self.weight)

        input_bin = binarize_conv2d(True)(input)

        out = nn.functional.conv2d(input_bin, weight_bin, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        '''if not self.bias is None:
            alpha = bin_w.alpha
            self.bias.org=self.bias.data.clone()
            self.bias.data = torch.sign(self.bias.data)
            #pdb.set_trace()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)#'''

        return out