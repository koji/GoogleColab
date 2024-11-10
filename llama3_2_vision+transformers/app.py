import torch
from PIL import Image
import gradio as gr
from transformers import MllamaForConditionalGeneration, AutoProcessor
import numpy as np

def load_model():
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        offload_folder="offload"
    )
    
    model.tie_weights()  # Tying weights for efficiency
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"mode is loaded on {device}")
    
    return model, processor



def run_process(text, image=None):
    model, processor = load_model()

    try:
        if image is not None:
            print('1')
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            image = image.convert("RGB").resize((224, 224))
            prompt = f"<|image|><|begin_of_text|>{text}"
            inputs = processor(images=[image], text=prompt, return_tensors="pt").to(model.device)
        else:
            prompt = f"<|begin_of_text|>{text}"
            inputs = processor(text=prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(**inputs, max_new_tokens=200)
        response = processor.decode(outputs[0], skip_special_tokens=True)

        return  response
    
    except Exception as e:
        print(f"error: {e}")
        return "something wrong. please try again after reload the app"

def create_interface():
    text_input = gr.Textbox(
        label="Describe your issue",
        placeholder="Describe the problem you're experiencing",
        lines=4,
    )
    
    image_input = gr.Image(label="upload an image")
    
    output = gr.Textbox(label="output", lines=5)
    
    
    interface = gr.Interface(
        fn=run_process,
        inputs=[text_input, image_input],
        outputs=output,
        title="multi modal app with llama3.2-vision",
        description="test llama3.2-vision with gradio"
    )
    
    return interface

def main():
    print("start multi-modal-app!")
    interface = create_interface()
    interface.launch(debug=True, share=True)


if __name__ == "__main__":
    main()
