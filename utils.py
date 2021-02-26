from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def time_torch_model(model, input, print_time=False):
    """
    Times the forward pass of a nn.Module with the given input.
    Inputs:
    ---------
        model: nn-Module
        input: torch.Tensor
        print_time: bool     Print the time?        (default: False)
    Output:
    ---------
        total_time_ms
    Author: Zeeshan Khan Suri
    """
    use_cuda = 'cuda' in input.device.type
    if use_cuda:
        torch.cuda.current_stream().synchronize()
    with torch.autograd.profiler.profile(use_cuda=use_cuda) as prof:
        model(input)
    total_time_ms = sum([item.cuda_time for item in prof.function_events])/1000
    if print_time:
        print("{:.3f} ms".format(total_time_ms))
    return total_time_ms

def tensor2numpy(x):
    """Converts torch tensor image to numpy image for plotting"""
    if isinstance(x, torch.Tensor):
        x=x.to('cpu').detach().numpy()
    x=np.array(x).squeeze()
    if len(x.shape)==3 and x.shape[0]==3:
        x=x.transpose((1,2,0))
    return x

def plot_input_output(img, output, print_output_shape=True, vmax=None, **kwargs):
    """
    Plots two images side-by-side.
    Inputs:
    -----------
        image1: PIL image or numpy array or torch.Tensor
        output: torch.Tensor or numpy array
        print_output_shape: bool
        **kwargs of matplotlib.pyplot.imshow()
    Author: Zeeshan Khan Suri
    """
    
    img,output = tensor2numpy(img), tensor2numpy(output)
    if vmax is None:
        vmax = np.percentile(output, 99)
    if print_output_shape: print(output.shape)

    fig, axes = plt.subplots(1,2, figsize=(20,3))
    axes[0].imshow(img)
    depthmap=axes[1].imshow(output,vmax=vmax, **kwargs)
    fig.colorbar(depthmap);
    
def scale_disp(disp, min_depth, max_depth):
    """
    Scale disparity such that depth lies in the range [min_depth, max_depth]
    Inputs
    -----------
    disp: 
        Disparity image
    min_depth:
        minimum depth
    max_depth:
        maximum depth
    Outputs
    -----------
    scaled_disp:
        Scaled dispartiy
    depth:
        1/scaled_disparity
    """
    min_disp, max_disp = 1 / max_depth, 1/min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return scaled_disp, 1/scaled_disp

def fuse_all_conv_bn(model:nn.Module):
    """
    Recursively fuses all pairs of Conv2d and BatchNorm2d of a nn.Module
    Usage:
        fuse_all_conv_bn(model)
    License: Copyright Zeeshan Khan Suri, 2021, MIT License
    """
    stack = []
    for name, module in model.named_children(): # immediate children
        if list(module.named_children()): # is not empty (not a leaf)
            fuse_all_conv_bn(module)
            
        if isinstance(module, nn.BatchNorm2d):
            if isinstance(stack[-1][1], nn.Conv2d):
                setattr(model, stack[-1][0], fuse_conv_bn_eval(stack[-1][1], module))
                setattr(model, name, nn.Identity())
        else:
            stack.append((name, module))

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Taken from https://stackoverflow.com/a/34325723/5984672
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
