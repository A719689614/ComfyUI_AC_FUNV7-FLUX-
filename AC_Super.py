from __future__ import annotations
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
from comfy.comfy_types import IO, ComfyNodeABC
import comfy.sd
import folder_paths
import random
import requests
from hashlib import md5
from .AC_FUN import CATEGORY


class UNETLoader(CATEGORY):
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False
        f1_taesd_enc = False
        f1_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
            elif v.startswith("taef1_encoder."):
                f1_taesd_dec = True
            elif v.startswith("taef1_decoder."):
                f1_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        if f1_taesd_dec and f1_taesd_enc:
            vaes.append("taef1")
        return vaes
    
    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes))
        decoder = next(filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes))

        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                              "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                              "clip_name1": (folder_paths.get_filename_list("text_encoders"), ),
                              "clip_name2": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["sdxl", "sd3", "flux", "hunyuan_video", "hidream"], ),
                              "vae_name": (s.vae_list(), )
                              },
                "optional": {
                              "device": (["default", "cpu"], {"advanced": True}),
                             }
                             }
    RETURN_TYPES = ("MODEL","CLIP", "VAE")
    FUNCTION = "load_unet"
  
    def load_unet(self, unet_name, weight_dtype, vae_name, clip_name1, clip_name2, type, device="default"):
        # VAE加载
        clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)

        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options=model_options)

        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        vae.throw_exception_if_invalid()

        # 模型加载
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model, clip, vae)



class CLIPTextEncode(ComfyNodeABC, CATEGORY):
    
    Condisiton_list = ["条件零化", "保持条件"]
    translate_list = ["zh","en"]
    MAX_RESOLUTION = 8192
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "negative": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."}),
                "translate_from": (s.translate_list,),
                "translate_to": (s.translate_list,),
                "apiid":  ('STRING', {'multiline': False}),
                "apikey":  ('STRING', {'multiline': False}),
                "translate":('BOOLEAN',{'default': False,}),
                "condition":(s.Condisiton_list,),
                "width": ("INT", {"default": 512, "min": 16, "max": s.MAX_RESOLUTION, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": s.MAX_RESOLUTION, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("Positive","Negative","Latent")
    FUNCTION = "encode"

    @classmethod
    def text_translation(self, positive,translate_from,translate_to, apiid, apikey):
        query = positive
        appid = apiid
        appkey = apikey
        from_lang = translate_from
        to_lang =  translate_to
        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        url = endpoint + path
        salt = random.randint(32768, 65536)
        sign = self.make_md5(appid + query + str(salt) + appkey)
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {'appid': appid, 'q': query, 'from': from_lang, 'to': to_lang, 'salt': salt, 'sign': sign}
        try:
            r = requests.post(url, params=payload, headers=headers)
            result = r.json()
            result = result['trans_result'][0]['dst']
        except:
            result = 'Error'
        
        return result

    @classmethod
    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return c
    def encode(self, clip, positive, translate_from,translate_to, apiid, apikey, negative, condition, translate,width, height, batch_size=1):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        
        if translate == True:
            tokens_1 = clip.tokenize(self.text_translation(positive,translate_from,translate_to, apiid, apikey))
        else:
            tokens_1 = clip.tokenize(positive)
        tokens_2 = clip.tokenize(negative)

        positive = clip.encode_from_tokens_scheduled(tokens_1)

        if condition == "条件零化":
            negative = self.zero_out( clip.encode_from_tokens_scheduled(tokens_2))
        if condition == "保持条件":
            negative = clip.encode_from_tokens_scheduled(tokens_2)
        
        # 返回潜空间设置
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)

        return (positive, negative, {"samples":latent})
  