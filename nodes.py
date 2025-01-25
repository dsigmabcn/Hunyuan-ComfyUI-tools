###### Starting document from https://github.com/welltop-cn/ComfyUI-TeaCache/blob/main/nodes.py #######


import math
import torch

from torch import Tensor
from unittest.mock import patch

from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.lightricks.model import precompute_freqs_cis
from comfy.ldm.common_dit import rms_norm

#### TEA CACHE ################
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

##### OOMPILE ##########################    
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
###               HYFETA ENHANCE CODE                                                    ###
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


#####################################################################
## NODE CLASSES MAPPING                                            ##
#####################################################################
NODE_CLASS_MAPPINGS = {
    #"TeaCacheForImgGen": TeaCacheForImgGen,
    #"TeaCacheForVidGen": TeaCacheForVidGen,
    "FetaEnhance": FetaEnhanceNode,
    "TeaCache_Hunyuan": TeaCache_Hunyuan,
    "CompileModel": CompileModel
    
}

NODE_DISPLAY_NAME_MAPPINGS = {k: v.TITLE for k, v in NODE_CLASS_MAPPINGS.items()}
