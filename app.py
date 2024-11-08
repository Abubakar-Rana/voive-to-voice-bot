import os
import numpy as np
import gradio as gr
from groq import Groq
import whisper
from TTS.api import TTS

# Set your Groq API key here
GROQ_API_KEY = "your-groq-api-key-here"  # Replace with your actual API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize models
def initialize_models():
    # Load Whisper model
    whisper_model = whisper.load_model("base")
    
    # Initialize Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)  # Using the API key directly
    
    # Initialize TTS
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
    
    return whisper_model, groq_client, tts


# Transcribe audio to text using Whisper
def transcribe_audio(audio_path, whisper_model):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Get response from Groq
def get_groq_response(client, input_text):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": input_text,
        }],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Convert text to speech
def text_to_speech(tts, text, output_path):
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path

# Main processing function
def process_audio(audio_input, whisper_model, groq_client, tts):
    try:
        # Save input audio temporarily
        input_path = "input_audio.wav"
        output_path = "output_audio.wav"
        
        # Step 1: Convert speech to text
        audio_input.save(input_path)  # Use save() method to save the audio file
        
        # Step 2: Convert speech to text
        input_text = transcribe_audio(input_path, whisper_model)
        
        # Step 3: Get LLM response
        response_text = get_groq_response(groq_client, input_text)
        
        # Step 4: Convert response to speech
        audio_output = text_to_speech(tts, response_text, output_path)
        
        return audio_output, input_text, response_text
        
    except Exception as e:
        return None, f"Error: {str(e)}", "An error occurred"

# Gradio interface
def create_interface():
    whisper_model, groq_client, tts = initialize_models()
    
    def wrapper(audio):
        return process_audio(audio, whisper_model, groq_client, tts)
    
    interface = gr.Interface(
        fn=wrapper,
        inputs=gr.Audio(type="filepath"),  # Corrected back to 'filepath' for Gradio's audio input
        outputs=[
            gr.Audio(label="Bot Response"),
            gr.Textbox(label="Transcribed Input"),
            gr.Textbox(label="Bot Response Text")
        ],
        title="Voice Chatbot",
        description="Speak to the chatbot and get voice responses back!",
        theme="default"
    )
    
    return interface

# For Google Colab and Hugging Face deployment
if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch()
