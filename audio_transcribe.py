# As a prerequisite, you must install Whisper from OpenAI GitHub repo
# This example skips this step
# In the local mode, WhisperTranscriber works on both a CPU and a GPU
# without any additional settings

from haystack.nodes.audio import WhisperTranscriber
whisper = WhisperTranscriber()
transcription = whisper.transcribe(audio_file="D:/LLM Bots/YouTube-Video-Summarization-App/test1.webm")