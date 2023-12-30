import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image
import torch

model_name="stabilityai/sdxl-turbo"
device="cuda"

pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to(device)

# Generate image
def generate_image(prompt, negative_prompt, seed, steps, scale):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(prompt=prompt, negative_prompt=negative_prompt, generator=generator, num_inference_steps=steps, guidance_scale=scale).images[0]
    return image

# Define input components
prompt_input = gr.Textbox(type="text", lines=5, label="Prompt")
negative_prompt_input = gr.Textbox(type="text", lines=5, label="Negative Prompt")
seed_input = gr.Number(label="Seed")

# Slider
steps_input = gr.Slider(minimum=1, maximum=10, step=1, value=1, interactive=True, label="Steps")
scale_input = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=0.0, interactive=True, label="Scale")

# Define output component
output = gr.Image(type="pil", label="Generated Image")

# Define Gradio Interface
simple_ui = gr.Interface(
    fn=generate_image,
    inputs=[prompt_input, negative_prompt_input, seed_input, steps_input, scale_input],
    outputs=output,
    title="SDXL Turbo via diffusers",
)

# Launch the interface
simple_ui.launch(share=True)
