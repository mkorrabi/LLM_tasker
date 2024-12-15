from typing import Optional
import gradio as gr

from src.custom_html_content import get_footer_html, get_header_html
from src.utils import (
    load_credentials,
)


class Interface:
    def __init__(self, title: str, auth_message: str) -> None:
        # attributs globaux utiles à l'interface
        self.title = title
        self.auth_message = auth_message

        # Init de l'interface
        self.build()

        # Execution
        # on préfère le faire en deux temps grâce à la méthode __call__() ?
        # self.launch(auth_message, server_name, server_port)

    def blocks(self):
        self.iface = gr.Blocks(
            analytics_enabled=False,
            title=self.title,
            theme=gr.themes.Default(
    primary_hue="violet",
),
            css="./apps/assets/css/main.css",
        )
        return self.iface

    def header(self):
        return gr.HTML(get_header_html(self.title))

    def footer(self):
        return gr.HTML(get_footer_html())

    def body(self):
        raise NotImplementedError()

    def build(self):
        with self.blocks() as block:
            self.header()
            self.body(block)
            self.footer()

    @property
    def allowed_paths(self):
        return ["./apps/assets/img/", "./assets/img/"]

    @property
    def blocked_paths(self):
        return [
            "README.md",
            "apps/main.py",
            "Dockerfile",
            "requirements-dev.txt",
            "requirements.txt",
            "pyproject.toml",
            "apps/src/",
        ]

    def launch(self, server_name: str, server_port: int):
        self.iface.launch(
            auth=load_credentials(),
            auth_message=self.auth_message,
            server_name=server_name,
            server_port=server_port,
            allowed_paths=self.allowed_paths,
            blocked_paths=self.blocked_paths,
            show_api=False,
            share=True,
        )

    def __call__(
        self, server_name: Optional[str] = None, server_port: Optional[int] = None
    ):
        self.launch(server_name=server_name, server_port=server_port)
