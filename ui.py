#!/usr/bin/env python3

import gradio as gr
from inference import infer_model


def main(audio_input):
    nsr, denoised = infer_model(audio_input)
    return nsr, denoised


ui = gr.Interface(
    fn=main,
    inputs=[
        gr.Audio(label="Upload or record audio", format="wav", type="filepath")
    ],
    outputs=gr.Audio(type="numpy"),
    live=True,
)

if __name__ == '__main__':
    ui.launch(show_api=False, share=True)
