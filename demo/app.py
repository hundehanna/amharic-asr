"""
Gradio demo for Amharic ASR using fine-tuned Whisper medium.
Deploy this to HuggingFace Spaces (Gradio SDK).
"""

import gradio as gr
from transformers import pipeline

MODEL_ID = "hundehanna/whisper-medium-amharic"  # <-- UPDATE after training

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    generate_kwargs={"language": "amharic", "task": "transcribe"},
)


def transcribe(audio_input):
    if audio_input is None:
        return "Please provide an audio file or record your voice."
    result = asr_pipe(audio_input)
    return result["text"]


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio"),
    outputs=gr.Textbox(label="Amharic Transcript (አማርኛ)", lines=4, show_copy_button=True),
    title="🎙️ Amharic Speech Recognition",
    description=(
        "Fine-tuned **whisper-medium** on the [Leyu Amharic dataset](https://leyu.ai/datasets). "
        "Record your voice or upload a `.wav`/`.mp3` file to get an Amharic transcript.\n\n"
        "**Language**: Amharic (አማርኛ) | **Model**: openai/whisper-medium (fine-tuned)"
    ),
    examples=[
        ["examples/example1.wav"],
        ["examples/example2.wav"],
        ["examples/example3.wav"],
    ],
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
