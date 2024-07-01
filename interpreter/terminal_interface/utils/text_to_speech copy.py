import logging
import os
import tempfile
import threading
import time

import simpleaudio as sa
from openai import OpenAI
from pydub import AudioSegment


# Wait until file exists
def download_and_write(audio_path, message, voice, speed, model):
    client = OpenAI()
    response = client.audio.speech.create(
        model=model or "tts-1",
        voice=voice or "onyx",
        input=message or "I can speak now?",
        speed=speed or "1.0",
    )

    with open(audio_path, "wb") as f:
        f.write(response.content)


def play_audio(audio_path):
    while not os.path.exists(audio_path):
        time.sleep(0.1)  # Wait until file exists

    while True:
        try:
            audio = AudioSegment.from_file(audio_path)
            if audio.duration_seconds > 0:  # Check if audio has content
                break
        except Exception as e:
            time.sleep(0.1)  # Wait and retry if there was an error loading

    play_obj = sa.play_buffer(
        audio.raw_data,
        num_channels=audio.channels,
        bytes_per_sample=audio.sample_width,
        sample_rate=audio.frame_rate,
    )
    play_obj.wait_done()
    os.remove(audio_path)  # Clean up the file after playing


def text_to_speech(message=None, voice=None, speed=None, model=None):
    # Create a unique temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(temp_fd)  # Close the file descriptor

    download_thread = threading.Thread(
        target=download_and_write, args=(temp_path, message, voice, speed, model)
    )
    play_thread = threading.Thread(target=play_audio, args=(temp_path,))

    download_thread.start()
    play_thread.start()

    download_thread.join()
    play_thread.join()


text_to_speech()
