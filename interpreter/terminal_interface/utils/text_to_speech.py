"""Module for text-to-speech functionality using OpenAI"""

import os
import queue
import subprocess
import tempfile
import threading

import simpleaudio as sa
from openai import OpenAI
from pydub import AudioSegment

# Queues for holding messages and paths to audio files
message_queue = queue.Queue()
playback_queue = queue.Queue()


def convert_opus_to_wav(opus_file_path):
    wav_file_path = opus_file_path.replace(".opus", ".wav")

    # Redirect output to null to avoid printing progress to the terminal
    command = f"ffmpeg -i {opus_file_path} {wav_file_path} > /dev/null 2>&1"
    os.system(command)  # Using os.system for simplicity in this example

    return wav_file_path


def audio_player():
    while True:
        wav_path = playback_queue.get()
        try:
            # print(f"Starting playback: {wav_path}")
            audio = AudioSegment.from_file(wav_path, format="wav")
            play_obj = sa.play_buffer(
                audio.raw_data,
                num_channels=audio.channels,
                bytes_per_sample=audio.sample_width,
                sample_rate=audio.frame_rate,
            )
            play_obj.wait_done()
            # print("Finished playing")
        except subprocess.CalledProcessError as e:
            print(e)
        finally:
            # os.remove(wav_path)

            playback_queue.task_done()


def audio_generator():
    while True:
        service, message = message_queue.get()

        if service == "openai":
            temp_path = generate_with_openai(message)
            playback_queue.put(convert_opus_to_wav(temp_path))
            message_queue.task_done()

        elif service == "piper":
            print("xUsing Piper")
            temp_path = generate_with_piper(message)
            print(temp_path)
            playback_queue.put(temp_path)
            message_queue.task_done()
        else:
            print(f"Unknown service: {service}")
            message_queue.task_done()

    return


def create_temp_file(suffix=".wav"):
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(temp_fd)
    return temp_path


def generate_with_piper(message):
    temp_path = create_temp_file()

    escaped_message = message  # shlex.quote(message)
    print("here")

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    piper_path = os.path.join(script_dir, "piper")
    model_path = os.path.join(script_dir, "piper", "en_US-lessac-medium.onnx")
    run_args = f"echo {escaped_message} | {piper_path} --model {model_path} --voice 'adam' --output_file '{temp_path}'  -q"

    print(run_args)

    output = subprocess.run(args=run_args, check=True, capture_output=True, shell=True)

    print(output.stdout.decode())

    return temp_path


def generate_with_openai(message):
    temp_path = create_temp_file(suffix=".opus")

    client = OpenAI()
    response = client.audio.speech.create(
        model="tts-1", response_format="opus", voice="alloy", input=message, speed="1.0"
    )

    with open(temp_path, "wb") as f:
        f.write(response.content)
        print("audio done")
        return temp_path


def text_to_speech(message, service):
    # Start the threads
    message_queue.put((service, message))


# Example usage
if __name__ == "__main__":
    # text_to_speech(message="Hello, this is test number one using open AI" ,service="openai" )
    text_to_speech(
        message="Hello, this is test number one using openai", service="piper"
    )

    threading.Thread(target=audio_generator, daemon=True).start()
    threading.Thread(target=audio_player, daemon=True).start()
