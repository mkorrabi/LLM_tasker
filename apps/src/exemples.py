import gradio as gr
from src.interface import Interface


class DemoInterface(Interface):
    def body(self):
        name = gr.Textbox(label="Name")
        output = gr.Textbox(label="Output Box")
        greet_btn = gr.Button("Hello!")

        # -----------------------------------------
        # Listeners
        # -----------------------------------------

        greet_btn.click(fn=self.greet, inputs=name, outputs=output, api_name="greet")

    def greet(self, name):
        return "Hello " + name + "!"
