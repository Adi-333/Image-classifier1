from transformers import pipeline
import gradio as gr

classifier = pipeline("image-classification", model="facebook/deit-base-distilled-patch16-224")

def classifyImage(image):
    res = classifier(image)
    outp = {r["label"]:r["score"] for r in res}
    return outp

iface = gr.Interface(
    fn= classifyImage,
    inputs=gr.Image(type="pil"),
    outputs="label",
    
)

iface.launch()