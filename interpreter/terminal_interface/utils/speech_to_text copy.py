import io
import wave

import numpy as np
import sounddevice as sd
from openai import OpenAI


class speech_to_text:
    def __init__(self, wake_word="Computer"):
        self.wake_word = wake_word.lower()

    ## self.api_key = api_key
    ##  client.api_key = self.api_key

    def record_audio(self, filename="recorded_audio.wav", duration=5):
        samplerate = 44100  # Sample rate
        myrecording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float64",
        )
        sd.wait()  # Wait until recording is finished
        myrecording_int = np.int16(myrecording * 32767)
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(myrecording_int)
        print(f"Audio recorded and saved to {filename}")
        return filename

    def transcribe_audio(self, file_path):
        client = OpenAI()
        with open(file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                file=audio, model="whisper-1", language="en"
            )
        print("Transcribed Text:", transcript)
        return transcript


# Usage
va = speech_to_text()
print("recording")
audio_file = va.record_audio()
transcribed_text = va.transcribe_audio(audio_file)
