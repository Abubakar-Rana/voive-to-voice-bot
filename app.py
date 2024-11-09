import os
import numpy as np
import gradio as gr
import whisper
from groq import Groq
from TTS.api import TTS
import tempfile
import soundfile as sf

# Configuration class
class Config:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
    WHISPER_MODEL = "base"
    TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
    SAMPLE_RATE = 16000

class VoiceChatbot:
    def __init__(self):
        print("Initializing Voice Chatbot...")
        self.config = Config()
        self.temp_dir = tempfile.mkdtemp()
        self.initialize_models()

    def initialize_models(self):
        """Initialize all required models"""
        try:
            # Initialize Whisper
            print("Loading Whisper model...")
            self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
            
            # Initialize Groq
            print("Initializing Groq client...")
            self.groq_client = Groq(api_key=self.config.GROQ_API_KEY)
            
            # Initialize TTS
            print("Loading TTS model...")
            self.tts = TTS(self.config.TTS_MODEL)
            
            print("All models initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            raise

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            result = self.whisper_model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            raise

    def get_llm_response(self, text):
        """Get response from Groq"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful and friendly AI assistant. Keep your responses concise and natural."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=150  # Keeping responses concise for better interaction
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            raise

    def text_to_speech(self, text):
        """Convert text to speech"""
        try:
            output_path = os.path.join(self.temp_dir, "response.wav")
            self.tts.tts_to_file(text=text, file_path=output_path)
            return output_path
        except Exception as e:
            print(f"TTS error: {e}")
            raise

    def process_audio(self, audio_input):
        """Main processing function"""
        try:
            if audio_input is None:
                return None, "No audio input received.", "Please provide audio input."

            # Save input audio temporarily
            input_path = os.path.join(self.temp_dir, "input_audio.wav")
            
            # Handle audio input
            if isinstance(audio_input, tuple):
                audio_data, sample_rate = audio_input
            else:
                input_path = audio_input
                audio_data, sample_rate = sf.read(input_path)

            # Ensure mono audio
            if len(audio_data.shape) == 2:
                audio_data = np.mean(audio_data, axis=1)

            # Save processed audio
            sf.write(input_path, audio_data, sample_rate)

            # Process the audio
            print("Processing audio...")
            print("1. Transcribing speech to text...")
            transcribed_text = self.transcribe_audio(input_path)
            print(f"Transcribed text: {transcribed_text}")

            print("2. Getting LLM response...")
            llm_response = self.get_llm_response(transcribed_text)
            print(f"LLM response: {llm_response}")

            print("3. Converting response to speech...")
            audio_output = self.text_to_speech(llm_response)
            print("Processing complete!")

            return audio_output, transcribed_text, llm_response

        except Exception as e:
            error_message = f"Error processing audio: {str(e)}"
            print(error_message)
            return None, error_message, "An error occurred"

def create_demo():
    """Create and configure the Gradio interface"""
    chatbot = VoiceChatbot()

    # Wrapper function for Gradio
    def process_wrapper(audio):
        return chatbot.process_audio(audio)

    # Create Gradio interface
    interface = gr.Interface(
        fn=process_wrapper,
        inputs=[
            gr.Audio(
                label="Voice Input",
                type="filepath",
                sources=["microphone", "upload"]
            )
        ],
        outputs=[
            gr.Audio(label="AI Response"),
            gr.Textbox(label="Transcribed Input"),
            gr.Textbox(label="AI Response Text")
        ],
        title="Real-time Voice Chatbot",
        description="""
        Speak or upload an audio file to chat with the AI assistant.
        The assistant will respond with voice and text.
        """,
        article="""
        How to use:
        1. Click the microphone icon to record your voice
        2. Speak clearly into your microphone
        3. Stop recording when finished
        4. Wait for the AI to process and respond
        5. Listen to the AI's voice response
        
        You can also upload pre-recorded audio files.
        """,
        examples=[],
        cache_examples=False
    )

    return interface

# For Google Colab
def launch_in_colab():
    demo = create_demo()
    demo.launch(debug=True, share=True)

# For Hugging Face Spaces
def launch_in_hf():
    demo = create_demo()
    demo.launch()

if __name__ == "__main__":
    import sys
    if 'google.colab' in sys.modules:
        launch_in_colab()
    else:
        launch_in_hf()
