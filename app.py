#!/usr/bin/env python

from __future__ import annotations

import functools
import sys

import gradio as gr
import huggingface_hub
import PIL.Image
import torch
import torch.nn as nn

sys.path.insert(0, 'Anime2Sketch')

from data import read_img_path, tensor_to_img
from model import UnetGenerator

TITLE = 'Anime2Sketch'
DESCRIPTION = 'https://github.com/Mukosame/Anime2Sketch'


def load_model(device: torch.device) -> nn.Module:
    norm_layer = functools.partial(nn.InstanceNorm2d,
                                   affine=False,
                                   track_running_stats=False)
    model = UnetGenerator(3,
                          1,
                          8,
                          64,
                          norm_layer=norm_layer,
                          use_dropout=False)

    path = huggingface_hub.hf_hub_download('public-data/Anime2Sketch',
                                           'netG.pth')
    ckpt = torch.load(path)
    for key in list(ckpt.keys()):
        if 'module.' in key:
            ckpt[key.replace('module.', '')] = ckpt[key]
            del ckpt[key]
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


@torch.inference_mode()
def run(image_file: str,
        model: nn.Module,
        device: torch.device,
        load_size: int = 512) -> PIL.Image.Image:
    tensor, orig_size = read_img_path(image_file, load_size)
    tensor = tensor.to(device)
    out = model(tensor)
    res = tensor_to_img(out)
    res = PIL.Image.fromarray(res)
    res = res.resize(orig_size, PIL.Image.Resampling.BICUBIC)
    return res


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = load_model(device)

fn = functools.partial(run, model=model, device=device)

examples = [['Anime2Sketch/test_samples/madoka.jpg']]

gr.Interface(
    fn=fn,
    inputs=gr.Image(label='Input', type='filepath'),
    outputs=gr.Image(label='Output'),
    examples=examples,
    title=TITLE,
    description=DESCRIPTION,
).queue().launch()
