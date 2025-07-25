from .AC_Super import UNETLoader, CLIPTextEncode

NODE_CLASS_MAPPINGS = {
    "AC_Super_UNET(FLUX)": UNETLoader,
    "AC_Super_CLIP(FLUX)": CLIPTextEncode,
}