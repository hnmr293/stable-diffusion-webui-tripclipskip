import os
from typing import Union

from modules import scripts
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img


def __set_value(p: StableDiffusionProcessing, script: type, value: Union[float,None], typ: Union[str,None]):
    if value is None and typ is None:
        return
    
    if value is not None:
        assert 1.0 <= value <= 12.0, f'invalid clip skip value: {value}'
    
    if typ is not None:
        assert typ in ['lerp', 'slerp'], f'invalid interpolation value: {typ}'
    
    args = list(p.script_args)
    
    if isinstance(p, StableDiffusionProcessingTxt2Img):
        all_scripts = scripts.scripts_txt2img.scripts
    else:
        all_scripts = scripts.scripts_img2img.scripts
    
    froms = [x.args_from for x in all_scripts if isinstance(x, script)]
    for idx in froms:
        assert idx is not None
        args[idx + 0] = True
        if value is not None:
            args[idx + 1] = value
        if typ is not None:
            args[idx + 2] = typ
    
    p.script_args = type(p.script_args)(args)
    

__init = False

def init_xyz(script: type):
    global __init
    
    if __init:
        return
    
    for data in scripts.scripts_data:
        name = os.path.basename(data.path)
        # t2i, i2i
        if name == 'xy_grid.py' or name == 'xyz_grid.py':
            #if script.is_txt2img:
            #    AxisOption = data.module.AxisOptionTxt2Img
            #else:
            #    AxisOption = data.module.AxisOptionImg2Img
            AxisOption = data.module.AxisOption
            v1 = AxisOption('Trip Clip Skip value', float, lambda p,x,xs: __set_value(p, script, x, None))
            v2 = AxisOption('Trip Clip Skip type', str, lambda p,x,xs: __set_value(p, script, None, x), choices=lambda: ['lerp', 'slerp'])
            data.module.axis_options.append(v1)
            data.module.axis_options.append(v2)
            
    __init = True
