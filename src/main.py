from model import get_recipe
import gradio as gr

demo = gr.ChatInterface(
  get_recipe,
  title="Recipes Generator",
  description="Ask recipes recommendations based on ingredients and instructions",
  examples=["Recommend a recipe with chicken", "A low-calorie chicken recipe", "What can I do with brocolli"],
)

if __name__ == "__main__":
  demo.launch()