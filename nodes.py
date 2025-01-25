###### Starting document from https://github.com/welltop-cn/ComfyUI-TeaCache/blob/main/nodes.py #######


import math
import torch

from torch import Tensor
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.common_dit import rms_norm


############## SEG ATTENTION ################################

#import math

from einops import rearrange
#import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_patcher
import comfy.samplers

###########################################################################################
###               TEA CACHE NODE                                                         ##
###########################################################################################
def poly1d(coefficients, x):
    '''
    This function is used later 
    '''
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result


def teacache_hunyuanvideo_forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        txt_mask: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor = None,
        control=None,
        transformer_options={},
    ) -> Tensor:
        patches_replace = transformer_options.get("patches_replace", {})
        rel_l1_thresh = transformer_options.get("rel_l1_thresh", {})

        initial_shape = list(img.shape)
        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

        if txt_mask is not None and not torch.is_floating_point(txt_mask):
            txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

        txt = self.txt_in(txt, timesteps, txt_mask)

        ids = torch.cat((img_ids, txt_ids), dim=1)
        pe = self.pe_embedder(ids)

        img_len = img.shape[1]
        if txt_mask is not None:
            attn_mask_len = img_len + txt.shape[1]
            attn_mask = torch.zeros((1, 1, attn_mask_len), dtype=img.dtype, device=img.device)
            attn_mask[:, 0, img_len:] = txt_mask
        else:
            attn_mask = None

        blocks_replace = patches_replace.get("dit", {})

        # enable teacache
        inp = img.clone()
        vec_ = vec.clone()
        img_mod1, _ = self.double_blocks[0].img_mod(vec_)
        modulated_inp = self.double_blocks[0].img_norm1(inp)
        modulated_inp = (1 + img_mod1.scale) * modulated_inp + img_mod1.shift

        if not hasattr(self, 'accumulated_rel_l1_distance'):
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            try:
                coefficients = [7.33226126e+02, -4.01131952e+02, 6.75869174e+01, -3.14987800e+00, 9.61237896e-02]
                self.accumulated_rel_l1_distance += poly1d(coefficients, ((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()))
                if self.accumulated_rel_l1_distance < rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            except:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = modulated_inp 

        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            for i, block in enumerate(self.double_blocks):
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"])
                        return out

                    out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
                    txt = out["txt"]
                    img = out["img"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            img += add

            img = torch.cat((img, txt), 1)

            for i, block in enumerate(self.single_blocks):
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, "attention_mask": attn_mask}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe, attn_mask=attn_mask)

                if control is not None: # Controlnet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            img[:, : img_len] += add

            img = img[:, : img_len]
            self.previous_residual = img - ori_img

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        shape = initial_shape[-3:]
        for i in range(len(shape)):
            shape[i] = shape[i] // self.patch_size[i]
        img = img.reshape([img.shape[0]] + shape + [self.out_channels] + self.patch_size)
        img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
        img = img.reshape(initial_shape)
        return img

class TeaCache_Hunyuan:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The video diffusion model the TeaCache will be applied to."}),
                #"model_type": (["hunyuan_video", "ltxv"],), #we only need hunyuan
                "rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "How strongly to cache the output of diffusion model. This value must be non-negative."})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_teacache"
    CATEGORY = "HUNYUAN TOOLS"
    TITLE = "TeaCache For Hunyuan"
    
    #def apply_teacache(self, model, model_type: str, rel_l1_thresh: float):
    def apply_teacache(self, model, rel_l1_thresh: float): #refined for only Hunyuan, we do not need the option model type
        if rel_l1_thresh == 0:
            return (model,)

        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        diffusion_model = new_model.get_model_object("diffusion_model")

        #remove model_type option
        forward_name = "forward_orig"
        replaced_forward_fn = teacache_hunyuanvideo_forward.__get__(
                            diffusion_model,
                            diffusion_model.__class__
                        )
  
        def unet_wrapper_function(model_function, kwargs):
            input = kwargs["input"]
            timestep = kwargs["timestep"]
            c = kwargs["c"]
            with patch.object(diffusion_model, forward_name, replaced_forward_fn):
                return model_function(input, timestep, **c)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)

        return (new_model,)

###########################################################################################
###              COMPILE NODE                                                            ##
###########################################################################################

class CompileModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the torch.compile will be applied to."}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "backend": (["inductor","cudagraphs", "eager", "aot_eager"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_compile"
    CATEGORY = "HUNYUAN TOOLS"
    TITLE = "Compile Model"
    
    def apply_compile(self, model, mode: str, backend: str, fullgraph: bool, dynamic: bool):
        new_model = model.clone()
        #clones the model to 'new_model', so we can apply _add_object_patch on it
        new_model.add_object_patch(
                                "diffusion_model",
                                torch.compile(
                                    new_model.get_model_object("diffusion_model"),
                                    mode=mode,
                                    backend=backend,
                                    fullgraph=fullgraph,
                                    dynamic=dynamic
                                )
                            )
        #torch.compile info here: https://pytorch.org/docs/main/generated/torch.compile.html
        return (new_model,)

###########################################################################################
###               HYFETA ENHANCE CODE                                                    ##
###########################################################################################
DEFAULT_ATTN = {
    'double': [i for i in range(0, 100, 1)],#[0,1,2,3,4,5,6,7,9,11,13,15,17,19,21,23,25],
    'single': [i for i in range(0, 100, 1)]
}

class FetaEnhanceNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "model": ("MODEL",),
            "feta_weight": ("FLOAT", {"default": 2, "min": -100.0, "max": 100.0, "step":0.01}),
        }, "optional": {
            "attn_override": ("ATTN_OVERRIDE",)
        }}
    RETURN_TYPES = ("MODEL",)

    CATEGORY = "HUNYUAN TOOLS"
    FUNCTION = "apply_feta_enchance"
    TITLE = "Feta Enhance Hunyuan"

    def apply_feta_enchance(self, model, feta_weight, attn_override=DEFAULT_ATTN):
        model = model.clone()

        model_options = model.model_options.copy()
        transformer_options = model_options['transformer_options'].copy()

        transformer_options['feta_weight'] = feta_weight
        transformer_options['feta_layers'] = attn_override
        model_options['transformer_options'] = transformer_options

        model.model_options = model_options
        return (model,)



###########################################################################################
###               STG ENHANCE CODE                                                    ###
###########################################################################################

class HunyuanSTG:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "stg_mode": (["STG-A", "STG-R"],),
                "stg_block_idx": ("INT", {"default": 0, "min": -1, "max": 39, "step": 1, "tooltip": "Block index to apply STG"}),
                "stg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Recommended values are ≤2.0"}),
                "stg_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Start percentage of the steps to apply STG"}),
                "stg_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "End percentage of the steps to apply STG"}),
            },
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_STG"
    CATEGORY = "HUNYUAN TOOLS"
    DESCRIPTION = "Spatio Temporal Guidance, https://github.com/junhahyung/STGuidance"
    TITLE = "STG Hunyuan"

    def apply_STG(self, model, stg_mode, stg_block_idx, stg_scale, stg_start_percent, stg_end_percent):#attn_override=DEFAULT_ATTN):
        model = model.clone()

        model_options = model.model_options.copy()
        transformer_options = model_options['transformer_options'].copy()

        transformer_options['stg_mode'] = stg_mode
        transformer_options['stg_block_idx'] = stg_block_idx
        transformer_options['stg_scale,'] = stg_scale,
        transformer_options['stg_start_percent'] = stg_start_percent
        transformer_options['stg_end_percent'] = stg_end_percent       
        #transformer_options['feta_layers'] = attn_override
        model_options['transformer_options'] = transformer_options

        model.model_options = model_options
        return (model,)


############## SEG ATTENTION ################################

import math

from einops import rearrange
import torch
import torch.nn.functional as F

from comfy.ldm.modules.attention import optimized_attention
import comfy.model_patcher
import comfy.samplers


def gaussian_blur_2d(img, kernel_size, sigma):
    '''
    This function is used later in SEGAttention
    '''
    height = img.shape[-1]
    kernel_size = min(kernel_size, height - (height % 2 - 1))
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])

    return img


class SEGAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                "blur": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 999.0, "step": 0.01, "round": 0.01}),
                "inf_blur": ("BOOLEAN", {"default": False} )
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "HUNYUAN TOOLS"
    DESCRIPTION = "SEG Attention node from  https://github.com/logtd/ComfyUI-SEGAttention"
    TITLE = "SEG Attention"

    def patch(self, model, scale, blur, inf_blur):
        m = model.clone()

        def seg_attention(q, k, v, extra_options, mask=None):
            _, sequence_length, _ = q.shape
            shape = extra_options['original_shape']
            oh, ow = shape[-2:]
            ratio = oh/ow
            d = sequence_length
            w = int((d/ratio)**(0.5))
            h = int(d/w)
            q = rearrange(q, 'b (h w) d -> b d w h', h=h)
            if not inf_blur:
                kernel_size = math.ceil(6 * blur) + 1 - math.ceil(6 * blur) % 2
                q = gaussian_blur_2d(q, kernel_size, blur)
            else:
                q = q.mean(dim=(-2, -1), keepdim=True)
            q = rearrange(q, 'b d w h -> b (h w) d')
            return optimized_attention(q, k, v, extra_options['n_heads'])

        def post_cfg_function(args):
            model = args["model"]

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]

            if scale == 0 or blur == 0:
                return uncond_pred + (cond_pred - uncond_pred)
            
            cond = args["cond"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]
            # Hack since comfy doesn't pass in conditionals and unconditionals to cfg_function
            # and doesn't pass in cond_scale to post_cfg_function
            len_conds = 1 if args.get('uncond', None) is None else 2 
            
            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, seg_attention, "attn1", "middle", 0)
            (seg,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            if len_conds == 1:
                return cond_pred + scale * (cond_pred - seg)

            return cond_pred + (scale-1.0) * (cond_pred - uncond_pred) + scale * (cond_pred - seg)

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)



#####################################################################
## NODE CLASSES MAPPING                                            ##
#####################################################################
NODE_CLASS_MAPPINGS = {
    #"TeaCacheForImgGen": TeaCacheForImgGen,
    #"TeaCacheForVidGen": TeaCacheForVidGen,
    "HunyuanSTG": HunyuanSTG,
    "FetaEnhance": FetaEnhanceNode,
    "TeaCache_Hunyuan": TeaCache_Hunyuan,
    "SEGAttention": SEGAttention,
    "CompileModel": CompileModel
    
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()} #it lloks for the title in the nodes, so it needs they need to be defined there, otherwise there is an error
