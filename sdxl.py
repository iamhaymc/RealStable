import gc
import time
import threading
from pathlib import Path
import numpy as np
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
)
from .ext import get_weighted_text_embeddings_sdxl
from pprint import pprint
from typing import Generator

# =================================================================================================


class SdxlPipe:

    def __init__(self):
        self.pipeline = None

    # ---------------------------------------------------------------------------------------------

    def load(
        self,
        checkpoint="sdxl-biglove-xl4",
        checkpoints_dir="./assets/models/sdxl-checkpoints",
        adapters_dir="./assets/models/sdxl-adapters",
        cache_dir="./cache/huggingface",
        local_only=True,
    ) -> "SdxlPipe":

        t0 = time.perf_counter()

        checkpoints_dir = Path(checkpoints_dir)
        adapters_dir = Path(adapters_dir)
        cache_dir = Path(cache_dir)

        if self.pipeline:
            self.free()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        has_gpu = device == "cuda"
        print(f"Using device: {device}")

        dtype = torch.bfloat16 if has_gpu else torch.float32
        print(f"Using data type: {dtype}")

        print(f"Loading checkpoint: {checkpoint}")
        model_file = checkpoints_dir.joinpath(checkpoint)
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            pretrained_model_name_or_path=model_file,
            local_files_only=local_only,
            use_safetensors=True,
            torch_dtype=dtype,
        ).to(device)

        print(f"Loading adapter: Lightning (8 Steps)")
        pipeline.load_lora_weights(
            adapters_dir,
            weight_name="sdxl-lora-lightning-s8.safetensors",
            local_files_only=local_only,
        )
        pipeline.fuse_lora(lora_scale=1.0)

        print(f"Assigning scheduler: Euler Ancestral")
        # Schedulers (https://huggingface.co/docs/diffusers/api/schedulers/overview)
        # - Euler Ancestral     -> EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", lower_order_final=True)
        # - DPM++ 2M SDE Karras -> DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True)
        # - LCM Exponential     -> LCMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", use_exponential_sigmas=True)
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
            lower_order_final=True,
        )

        pipeline.enable_xformers_memory_efficient_attention()

        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"Load duration: {dt:.6f} secs")

        self.pipeline = pipeline

        return self

    # ---------------------------------------------------------------------------------------------

    def free(self) -> "SdxlPipe":
        # Verify resources
        if not self.pipeline:
            raise Exception("Pipeline is not instantiated")

        # Dispose resources
        print(f"Disposing pipeline resources")
        del self.pipeline
        self.pipeline = None
        gc.collect()
        torch.cuda.empty_cache()

        return self

    # ---------------------------------------------------------------------------------------------

    def call(
        self, prompt_pos, prompt_neg, seed=None, clip_skip=2
    ) -> Generator[str, None, None]:

        # Verify resources
        if not self.pipeline:
            raise Exception("Pipeline is not instantiated")

        t0 = time.perf_counter()

        # Seed generator
        if not seed:
            seed = np.random.randint(0, 2**32)
        generator = torch.Generator(self.pipeline.device).manual_seed(seed)
        print(f"Using seed: {seed}")

        prompt_embeds = get_weighted_text_embeddings_sdxl(
            self.pipeline, prompt=prompt_pos, neg_prompt=prompt_neg, clip_skip=clip_skip
        )

        # Include hyperparameters
        genkwargs = dict(
            guidance_scale=1.0,
            num_inference_steps=8,
            clip_skip=clip_skip,
            generator=generator,
            width=832,
            height=1216,
            out_type="pil",
        )
        print("Hyperparameters:")
        pprint(genkwargs)

        # Include non-hyperparameters
        genkwargs["prompt_embeds"] = (prompt_embeds[0],)
        genkwargs["negative_prompt_embeds"] = (prompt_embeds[1],)
        genkwargs["pooled_prompt_embeds"] = (prompt_embeds[2],)
        genkwargs["negative_pooled_prompt_embeds"] = (prompt_embeds[3],)

        # Generate response
        print(f"Generating images...")
        with torch.inference_mode():
            images = self.pipeline(**genkwargs).images

        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"Generation duration: {dt:.6f} secs")

        for image in images:
            yield image
