import wave

import numpy as np
import piper
import sounddevice as sd
from openai import OpenAI


class SpeechToText:
    """A class for speech-to-text conversion."""

    def __init__(self, wake_word="Computer"):
        """Initialize the SpeechToText class."""
        self.wake_word = wake_word.lower()

    def record_audio(self, filename="recorded_audio.wav", duration=4):
        """Record audio and save it to a file."""
        samplerate = 44100  # Sample rate
        myrecording = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=1,
            dtype="float64",
        )
        sd.wait()  # Wait until recording is finished
        myrecording_int = np.int16(myrecording * 32767)
        with wave.open(filename, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(myrecording_int.tobytes())
        print(f"Audio recorded and saved to {filename}")
        return filename

    def transcribe_audio_openai(self, file_path):
        """Transcribe audio using the OpenAI API."""
        client = OpenAI()
        with open(file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                file=audio, model="whisper-1", language="en"
            )
        print("Transcribed Text (OpenAI):", transcript)
        return transcript

    def transcribe_audio_piper(self, file_path):
        """Transcribe audio using the Piper library."""
        try:
            with open(file_path, "rb") as audio:
                transcript = piper.transcribe(audio)
            print("Transcribed Text (Piper):", transcript)
            return transcript
        except AttributeError:
            print("Piper module is not available.")
            return None


# Usage
va = SpeechToText()
print("Recording...")
AUDIO_FILE = va.record_audio()
transcribed_text_openai = va.transcribe_audio_openai(AUDIO_FILE)
transcribed_text_piper = va.transcribe_audio_piper(AUDIO_FILE)
