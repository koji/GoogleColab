import gradio as gr
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
)
from safetensors.torch import load_file

model_name = "stabilityai/sdxl-turbo"
custom_model_path = "t-shirt_design-sdxl.safetensors"
device = "cuda"

# Load the base SDXL Turbo model
pipe = StableDiffusionXLPipeline.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to(device)

# Load and set the custom weights
custom_weights = load_file(custom_model_path)
pipe.unet.load_state_dict(custom_weights, strict=False)

# Define samplers
def create_karras_scheduler(scheduler_class):
    return scheduler_class.from_config(pipe.scheduler.config, use_karras_sigmas=True)

samplers = {
    "DPM++ 2M": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    "DPM++ 2M Karras": create_karras_scheduler(DPMSolverMultistepScheduler),
    "Euler": EulerDiscreteScheduler.from_config(pipe.scheduler.config),
    "Euler a": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
    "DDIM": DDIMScheduler.from_config(pipe.scheduler.config),
}

def generate_image(prompt, negative_prompt, seed, steps, scale, sampler, clip_skip):
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Set the selected sampler
    pipe.scheduler = samplers[sampler]
    
    # Set CLIP skip
    pipe.text_encoder.num_hidden_layers = 22 - clip_skip  # SDXL has 22 layers total

    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        generator=generator, 
        num_inference_steps=steps, 
        guidance_scale=scale
    ).images[0]
    
    # Reset CLIP skip to default
    pipe.text_encoder.num_hidden_layers = 22

    return image

# Define input components
prompt_input = gr.Textbox(type="text", lines=5, label="Prompt")
negative_prompt_input = gr.Textbox(type="text", lines=5, label="Negative Prompt")
seed_input = gr.Number(label="Seed", value=42)
steps_input = gr.Slider(minimum=1, maximum=50, step=1, value=20, interactive=True, label="Steps")
scale_input = gr.Slider(minimum=0.0, maximum=20.0, step=0.1, value=7.5, interactive=True, label="Guidance Scale")
sampler_input = gr.Dropdown(choices=list(samplers.keys()), value="DPM++ 2M", label="Sampler")
clip_skip_input = gr.Slider(minimum=1, maximum=4, step=1, value=1, interactive=True, label="CLIP Skip")

# Define output component
output = gr.Image(type="pil", label="Generated Image")

# Define Gradio Interface
simple_ui = gr.Interface(
    fn=generate_image,
    inputs=[prompt_input, negative_prompt_input, seed_input, steps_input, scale_input, sampler_input, clip_skip_input],
    outputs=output,
    title="Custom SDXL Turbo Model",
)

# Launch the interface
simple_ui.launch(share=True)
