from transformers import AutoProcessor,BarkModel
import scipy

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained('suno/bark')
model.to('cpu')


def generate_audio(text,preset,output):
  inputs = processor(text,voice_preset=preset)
  for k,v in inputs.items():
    inputs[k] =v.to('cpu')
  audio_array = model.generate(**inputs)
  audio_array = audio_array.cpu().numpy().squeeze()
  sample_rate = model.generation_config.sample_rate
  scipy.io.wavfile.write(output,rate=sample_rate,data=audio_array)


generate_audio(text='The video discusses about -um- Type2 Diabetes Detection using Traditional ML Algorithms',
               preset = "v2/hi_speaker_2",
               output='synthesize1.wav')
