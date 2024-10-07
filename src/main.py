from model import get_recipe
import gradio as gr

demo = gr.Interface(
    fn=get_recipe,
    inputs=["text"],
    outputs=["text"],
)

if __name__ == "__main__":
  demo.launch(share=True)