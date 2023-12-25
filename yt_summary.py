import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time
from transformers import AutoProcessor,BarkModel
import scipy
import base64

st.set_page_config(
    layout="wide",
    page_title='Youtube Summary'
)

def download_video(url):
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

def initialize_model(full_path):
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )
#Chains(LangChain) -> Node(haystack)

def initialize_prompt_node(model):
    summary_prompt = "deepset/summarization" # using default prompt of haystack
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])

    output = pipeline.run(file_paths=[file_path])
    print(output["results"])
    return output


processor = AutoProcessor.from_pretrained("suno/bark")
model_bark = BarkModel.from_pretrained('suno/bark')
model_bark.to('cpu')


def generate_audio(text,preset,output):
  inputs = processor(text,voice_preset=preset)
  for k,v in inputs.items():
    inputs[k] =v.to('cpu')
  audio_array = model_bark.generate(**inputs)
  audio_array = audio_array.cpu().numpy().squeeze()
  sample_rate = model_bark.generation_config.sample_rate
  scipy.io.wavfile.write(output,rate=sample_rate,data=audio_array)

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )


def main():

    # Set the title and background color
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with the Llama 2 🦙,Whisper, Haystack, Streamlit❤️ and Bark🐶')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("This app allows you to summarize while watching a YouTube video.")
        st.write("Enter a YouTube URL in the input box below and click 'Submit' to start. This app is built by AI Anytime.")

    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()  # Start the timer
        # Download video
        file_path = "D:/LLM Bots/YouTube-Video-Summarization-App/test1.webm"

        # Initialize model
        full_path = "D:/LLM Bots/YouTube-Video-Summarization-App/llama-2-7b-32k-instruct.Q2_K.gguf"
        model = initialize_model(full_path)
        prompt_node = prompt_node = initialize_prompt_node(model)
        # Transcribe audio
        output = transcribe_audio(file_path, prompt_node)

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = st.columns([1,1])

        # Column 1: Video view
        with col1:
            st.video(youtube_url)

        # Column 2: Summary View
        with col2:
            st.header("Summarization of YouTube Video")
            st.write(output)
            st.success(output["results"][0].split("\n\n[INST]")[0])
            st.write(f"Time taken: {elapsed_time:.2f} seconds")
        speech_convert= output["results"][0].split("\n\n[INST]")[0]

        generate_audio(text= speech_convert,
                   preset = "v2/hi_speaker_2",
                   output='synthesize2.wav')
    



        st.write("Playing Summarisation Audio with Bark!")

        autoplay_audio("D:/LLM Bots/YouTube-Video-Summarization-App/synthesize2.wav")

if __name__ == "__main__":
    main()