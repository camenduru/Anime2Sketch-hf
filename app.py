#!/usr/bin/env python

from __future__ import annotations

import argparse
import functools
import os
import sys

import gradio as gr
import huggingface_hub
import PIL.Image
import torch
import torch.nn as nn

sys.path.insert(0, 'Anime2Sketch')

from data import read_img_path, tensor_to_img
from model import UnetGenerator

TITLE = 'Mukosame/Anime2Sketch'
DESCRIPTION = 'This is a demo for https://github.com/Mukosame/Anime2Sketch.'
ARTICLE = None

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    parser.add_argument('--allow-flagging', type=str, default='never')
    return parser.parse_args()


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

    path = huggingface_hub.hf_hub_download('hysts/Anime2Sketch',
                                           'netG.pth',
                                           use_auth_token=TOKEN)
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
def run(image_file,
        model: nn.Module,
        device: torch.device,
        load_size: int = 512) -> PIL.Image.Image:
    tensor, orig_size = read_img_path(image_file.name, load_size)
    tensor = tensor.to(device)
    out = model(tensor)
    res = tensor_to_img(out)
    res = PIL.Image.fromarray(res)
    res = res.resize(orig_size, PIL.Image.Resampling.BICUBIC)
    return res


def main():
    gr.close_all()

    args = parse_args()
    device = torch.device(args.device)

    model = load_model(device)

    func = functools.partial(run, model=model, device=device)
    func = functools.update_wrapper(func, run)

    examples = [['Anime2Sketch/test_samples/madoka.jpg']]

    gr.Interface(
        func,
        gr.inputs.Image(type='file', label='Input'),
        gr.outputs.Image(type='pil', label='Output'),
        examples=examples,
        title=TITLE,
        description=DESCRIPTION,
        article=ARTICLE,
        theme=args.theme,
        allow_flagging=args.allow_flagging,
        live=args.live,
    ).launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
