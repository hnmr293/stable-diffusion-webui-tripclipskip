import math
from typing import Union

import gradio as gr

import torch
from torch import nn, Tensor
from modules.processing import process_images, fix_seed, Processed, StableDiffusionProcessing
from modules import scripts
from modules.shared import opts

from scripts.xyz import init_xyz
from scripts.sdhook import SDHook

NAME = 'TripClipSkip'

def E(msg: str):
    return f'[{NAME}] {msg}'


def slerp(val, low, high):
    # cf. https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res


class ClipHooker(SDHook):
    
    def __init__(self, enabled: bool, value: float, slerp: bool):
        super().__init__(enabled)
        
        value = float(value)
        assert 1.0 <= value <= 12.0
        
        self.slerp = slerp
        self.v1 = math.floor(value)
        self.v2 = math.ceil(value)
        if self.v2 == 12:
            self.v1 = 11
            self.v2 = 12
        elif self.v1 == self.v2:
            self.v2 += 1
        self.r = value - self.v1 # 0..1
    
    def hook_clip(self, p: StableDiffusionProcessing, clip: nn.Module):
        skip = False
        def hook(module, inputs, output):
            # ignore original inputs and output
            nonlocal skip
            
            if skip:
                return output
            
            org = opts.CLIP_stop_at_last_layers
            skip = True
            try:
                opts.CLIP_stop_at_last_layers = self.v1
                x1 = module(inputs[0])
                opts.CLIP_stop_at_last_layers = self.v2
                x2 = module(inputs[0])
                
                if self.slerp:
                    x = slerp(self.r, x1, x2)
                else:
                    x = torch.lerp(x1, x2, self.r)
                    
                return x
            finally:
                opts.CLIP_stop_at_last_layers = org
                skip = False
            
        self.hook_layer(clip, hook)
    

class Script(scripts.Script):
    
    def __init__(self):
        super().__init__()
        self.last_hooker: Union[ClipHooker,None] = None

    def title(self):
        return NAME
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def ui(self, is_img2img):
        mode = 'img2img' if is_img2img else 'txt2img'
        id = lambda x: f'{NAME.lower()}-{mode}-{x}'
        
        with gr.Group():
            with gr.Accordion('Trip Clip Skip', open=False):
                enabled = gr.Checkbox(label='Enable', value=False)
                clipskip = gr.Slider(minimum=1.0, maximum=12.0, value=2.0, step=0.01, label='Clip skip', elem_id=id('value'))
                typ = gr.Radio(choices=['lerp', 'slerp'], value='lerp', label='Interpolation type', elem_id=id('interpolation'))
        
        return [enabled, clipskip, typ]
    
    def process(self, p: StableDiffusionProcessing, enabled: bool, v: float, typ: str):
        def restore():
            if self.last_hooker is not None:
                self.last_hooker.__exit__(None, None, None)
                self.last_hooker = None
        
        restore()
        
        if not enabled:
            return
        
        v = float(v)
        assert typ in ['lerp', 'slerp']
        self.last_hooker = ClipHooker(True, v, typ == 'slerp')
        self.last_hooker.setup(p)
        self.last_hooker.__enter__()
        
        p.extra_generation_params.update({
            f'{NAME}_value': v,
            f'{NAME}_type': typ,
        })

init_xyz(Script)
