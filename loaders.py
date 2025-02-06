from .common import SeedContext, SEED_INPUT
import random
from .randoms import RandomBase

from nodes import LoraLoader, CheckpointLoaderSimple
from PIL import Image, ImageOps
import numpy as np
import torch
import os

from folder_paths import folder_names_and_paths, get_folder_paths
from comfy.sd import load_checkpoint_guess_config

class RandomLoaderException(Exception):
    pass

class KeepForRandomBase(RandomBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
            "seed": SEED_INPUT(), 
            "keep_for": ("INT", {"default": 1, "min": 1, "max": 100}), 
            "mode": ( ["random", "systematic"], {}), 
            "follow_subfolders": (["false", "true"], {"default": "false"}),
            "subfolder": ("STRING", {"default": "random"}) 
        }}
    
    @classmethod
    def add_input_types(cls, it):
        add = KeepForRandomBase.INPUT_TYPES()
        it['required'].update(add['required'])
        return it

    def __init__(self):
        self.since_last_change = 0
        self.last_systematic = -1
        self.result = None
        self.systematic = False
        self.follow_subfolders = False

    def func(self, seed, keep_for, mode, subfolder, follow_subfolders, **kwargs):
        self.subfolder = subfolder
        self.since_last_change += 1
        self.follow_subfolders = follow_subfolders == "true"
        
        self.systematic = (mode == "systematic")
        if self.since_last_change >= keep_for or self.result is None:
            self.since_last_change = 0
            with SeedContext(seed):
                self.result = self.func_(**kwargs)
        return self.result
    
    def _get_list(self, category):
        if category not in folder_names_and_paths:
            raise RandomLoaderException(f"Invalid category: {category}")
        
        fnap = folder_names_and_paths[category]
        options = set()

        for folder in fnap[0]:
            search_folder = os.path.join(folder, self.subfolder)
            if os.path.exists(search_folder):
                if self.follow_subfolders:
                    for root, _, files in os.walk(search_folder):
                        options.update(
                            os.path.join(root, file) for file in files if os.path.splitext(file)[1] in fnap[1]
                        )
                else:
                    options.update(
                        os.path.join(search_folder, file) for file in os.listdir(search_folder)
                        if os.path.isfile(os.path.join(search_folder, file)) and os.path.splitext(file)[1] in fnap[1]
                    )
        
        if not options:
            raise RandomLoaderException(f"No files found in {search_folder}")
        return list(options)
    
    def choose_from(self, category_or_list):
        lst = self._get_list(category_or_list) if isinstance(category_or_list, str) else category_or_list
        if not lst:
            raise RandomLoaderException("No items available to choose from")
        
        if self.systematic:
            self.last_systematic = (self.last_systematic + 1) % len(lst)
            return lst[self.last_systematic]
        return random.choice(lst)

class LoadRandomCheckpoint(KeepForRandomBase, CheckpointLoaderSimple):
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("model", "CLIP", "VAE", "ckpt_name",)

    def func_(self):
        ckpt_path = self.choose_from("checkpoints")
        print(f"[INFO] Selected Checkpoint: {ckpt_path}")
        out = load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=get_folder_paths("embeddings"))
        return out[:3] + (os.path.splitext(os.path.basename(ckpt_path))[0],)
    
class LoadRandomLora(KeepForRandomBase, LoraLoader):
    RETURN_TYPES = ("MODEL", "CLIP", "STRING",)
    RETURN_NAMES = ("model", "clip", "lora_name",)

    def __init__(self):
        LoraLoader.__init__(self)
        KeepForRandomBase.__init__(self)

    @classmethod
    def INPUT_TYPES(cls):
        it = LoraLoader.INPUT_TYPES()
        it['required'].pop('lora_name')
        return cls.add_input_types(it)

    def func_(self, **kwargs):
        lora_name = self.choose_from("loras")
        lora_name = os.path.join(self.subfolder, os.path.basename(lora_name))
        print(f"[INFO] Selected Lora: {lora_name}")
        return self.load_lora(lora_name=lora_name, **kwargs) + (os.path.splitext(os.path.basename(lora_name))[0],)
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

class LoadRandomImage(KeepForRandomBase):
    @classmethod
    def INPUT_TYPES(cls):
        it = {'required': {"folder":("STRING", {"default": ""}), "extensions":("STRING", {"default": ".png, .jpg, .jpeg"})}}
        return cls.add_input_types(it)

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("image", "filepath",)

    def get_filenames(self, folder: str, extensions: str):
        image_extensions = {e.strip().lower() for e in extensions.split(",")}
        return [file for file in os.listdir(folder) if os.path.splitext(file)[1].lower() in image_extensions]

    def func_(self, folder: str, extensions: str):
        filenames = self.get_filenames(folder, extensions)
        if not filenames:
            raise RandomLoaderException(f"No images found in folder: {folder}")
        
        filename = self.choose_from(filenames)
        filepath = os.path.join(folder, filename)
        print(f"[INFO] Selected Image: {filepath}")
        
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img).convert("RGB")
        image = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return image, filepath
