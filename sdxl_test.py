import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, LCMScheduler
from safetensors.torch import load_file

from pylib import get_weighted_text_embeddings_sdxl
# -----------------------------------------------------------------------------

prompt_positive = "pussy, woman, smiling, blond hair, cowboy shot, front view, black body pulled-aside, cameltoe pussy, shaved vulva, beach, on allfours, looking back, cowboy hat, closeup, from below, Dutch angle, cum in pussy, cum overflow"
prompt_negative = ""

# -----------------------------------------------------------------------------

# vae = AutoencoderKL.from_pretrained(
#     "./dat/models/sdxl-vae-fp16fix",
#     torch_dtype=torch.float16,
#     local_files_only=True,
# )

# unet = UNet2DConditionModel.from_config(
#     "./dat/models/sdxl-chkpt-biglove", subfolder="unet",
#     use_safetensors=True,
#     local_files_only=True,
#     ).to("cuda", torch.float16)
# unet.load_state_dict(load_file(
#     "./dat/models/sdxl-lora-dmd2/sdxl-unet-dmd2_4step_fp16.safetensors",
#     device="cuda"
#     ))
# unet.load_state_dict(load_file(
#     "./dat/models/sdxl-lightning/sdxl-lightningunet-dmd2_4step_fp16.safetensors",
#     device="cuda"
#     ))

pipe = StableDiffusionXLPipeline.from_pretrained(
    "./dat/models/sdxl-chkpt-biglove",
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=True,
    #unet=unet
    #vae=vae,
).to("cuda")

# pipe.load_lora_weights(
#     "./dat/models/sdxl-lora-dmd2",
#     weight_name="sdxl-lora-dmd2_4step_fp16.safetensors",
#     local_files_only=True,
# )
pipe.load_lora_weights(
    "./dat/models/sdxl-lightning",
    weight_name="sdxl-lora-lightning-8step.safetensors",
    local_files_only=True,
)
pipe.fuse_lora(lora_scale=1.0)

# LCM Exponential
# See: https://huggingface.co/docs/diffusers/api/schedulers/overview
# pipe.scheduler = LCMScheduler.from_config(
#     pipe.scheduler.config, 
#     timestep_spacing="trailing",
#     use_exponential_sigmas=True,
# )
# See: https://huggingface.co/docs/diffusers/api/schedulers/overview
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipe.scheduler.config, 
    timestep_spacing="trailing",
    lower_order_final=True
)
# DPM++ 2M SDE Karras 
# See: https://huggingface.co/docs/diffusers/api/schedulers/overview
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#     pipe.scheduler.config,
#     algorithm_type = "sde-dpmsolver++",
#     use_karras_sigmas=True
# )

pipe.enable_xformers_memory_efficient_attention()
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_slicing()

# -----------------------------------------------------------------------------

CLIP_SKIP = 2

seed = 696613763  # np.random.randint(0, np.iinfo(np.int32).max)
generator = torch.Generator(pipe.device).manual_seed(seed)

prompt_embeds = get_weighted_text_embeddings_sdxl(
    pipe, prompt=prompt_positive, neg_prompt=prompt_negative, clip_skip=CLIP_SKIP
)

image = pipe(
    prompt_embeds=prompt_embeds[0],
    negative_prompt_embeds=prompt_embeds[1],
    pooled_prompt_embeds=prompt_embeds[2],
    negative_pooled_prompt_embeds=prompt_embeds[3],
    guidance_scale=1.0,
    num_inference_steps=8,
    clip_skip=CLIP_SKIP,
    generator=generator,
    width=832,
    height=1216,
    out_type="pil",
).images[0]

image.save("output.png")




# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import numpy as np
# import torch
# from diffusers import (
#     StableDiffusionXLPipeline,
#     EulerAncestralDiscreteScheduler,
#     DPMSolverSinglestepScheduler,
#     DPMSolverMultistepScheduler,
#     LCMScheduler,
# )
# from safetensors.torch import load_file

# from pylib import get_weighted_text_embeddings_sdxl

# # -----------------------------------------------------------------------------

# prompt_positive = "score_9_up, photo \(medium\), paid reward, photorealistic, realistic, Olivia, 18 year old woman, white woman, tanned skin, thin body, twintails, amateur photo, mirror selfie, (film grain:1.3), (face out of frame:1.2), lying down on sofa, vagina visible, foreground lighting, warm lighting on lower body, (iphone in her hand, iphone in front of her face, looking at her phone, fingering herself, wet pussy, dripping:1.2), low angle shot, wide-angle lens perspective, dramatic upward view, behind view, worms eye view, dorm room window ledge, cluttered with small potted plants, collected knick-knacks, illuminated faintly by external campus lights filtering through dirty window pane, silhouetted against dim room interior , Night, interior of ancient Roman amphitheater, crumbling stone arches, tiered seating rings surrounding sandy arena floor, bright sunlight entering through openings"
# prompt_negative = "eyes open,"

# # -----------------------------------------------------------------------------

# # vae = AutoencoderKL.from_pretrained(
# #     "./dat/models/sdxl-vae-fp16fix",
# #     torch_dtype=torch.float16,
# #     local_files_only=True,
# # )

# # unet = UNet2DConditionModel.from_config(
# #     "./dat/models/sdxl-chkpt-biglove", subfolder="unet",
# #     use_safetensors=True,
# #     local_files_only=True,
# #     ).to("cuda", torch.float16)
# # unet.load_state_dict(load_file(
# #     "./dat/models/sdxl-lora-dmd2/sdxl-unet-dmd2_4step_fp16.safetensors",
# #     device="cuda"
# #     ))
# # unet.load_state_dict(load_file(
# #     "./dat/models/sdxl-lightning/sdxl-lightningunet-dmd2_4step_fp16.safetensors",
# #     device="cuda"
# #     ))

# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "B:/assets/models/sdxl-checkpoints/sdxl-chkpt-lustify-olt",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     local_files_only=True,
# ).to("cuda")

# pipe.load_lora_weights(
#     "B:/assets/models/sdxl-adapters",
#     weight_name="sdxl-lora-lightning-s4.safetensors",
#     local_files_only=True,
# )
# pipe.fuse_lora(lora_scale=1.0)

# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
#     pipe.scheduler.config, timestep_spacing="trailing", lower_order_final=True
# )

# pipe.enable_xformers_memory_efficient_attention()

# # -----------------------------------------------------------------------------

# CLIP_SKIP = 2

# seed = 531460114079232  # np.random.randint(0, np.iinfo(np.int32).max)
# generator = torch.Generator(pipe.device).manual_seed(seed)

# prompt_embeds = get_weighted_text_embeddings_sdxl(
#     pipe, prompt=prompt_positive, neg_prompt=prompt_negative, clip_skip=CLIP_SKIP
# )

# image = pipe(
#     prompt_embeds=prompt_embeds[0],
#     negative_prompt_embeds=prompt_embeds[1],
#     pooled_prompt_embeds=prompt_embeds[2],
#     negative_pooled_prompt_embeds=prompt_embeds[3],
#     guidance_scale=1.0,
#     num_inference_steps=8,
#     clip_skip=CLIP_SKIP,
#     generator=generator,
#     width=832,
#     height=1216,
#     out_type="pil",
# ).images[0]

# image.save("output.png")
