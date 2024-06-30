from pytube import YouTube
import os

import sys

import openai

from load_tts_model import load_tts_model
import wave
import re

#chucking video:
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import math

#chucking words of over 3000 tokens:
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from pyannote.audio import Pipeline
# Use your HF key
with open("HF_token.txt") as f:
	HF_key = f.read().strip()	
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_key)

# send pipeline to GPU (when available)
#import torch
#pipeline.to(torch.device("cuda"))


# Use your own API key
with open("openai-api-key.txt") as f:
	openai.api_key = f.read().strip()	

transcript = []

conversation_history = []

bot_response = None

prompt = None

filepath = None

current_filepath = None

two_people = False

############# setup and load TTS-model #####################
syn, syn2 = load_tts_model()
############################################################

#For generating the transcript with whisper
def transcribe_video(filepath):

    print("file =")
    print(filepath)
    video = VideoFileClip(filepath)
    audio = video.audio

    mp3_file = "./mp3-files/audio.mp3"
    wav_file = "./wav-files/audio.wav"
    audio.write_audiofile(mp3_file)

    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

    audio = wave.open(wav_file, mode='rb')

    # apply pretrained pipeline
    diarization = pipeline(wav_file)

    # print the result
    # TODO: make array of spekaers = [speaker1, speaker2, speaker1, ...] and their times
    # times = [[start stop], [start stop], ....]
    # then make new wav files for speaker1, speaker2, speaker1...etc
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    segment_duration = 30  # seconds
    print("Video duration = ", video.duration)
    transcripts = []
    num_segments = math.ceil(video.duration / segment_duration)
    print("Number of segments =", num_segments)

    # Loop through the segments
    for i in range(num_segments):
        # Calculate the start and end times for the current segment
        print("Transcribing segment nr: ", i+1)
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, video.duration)
        segment = video.subclip(start_time, end_time)
        print("Audio-segment duration: ", segment.duration)
        segment_name = f"segment_{i+1}.mp3"
        segment.audio.write_audiofile(segment_name)

        # Pass the audio segment to WISPR for speech recognition
        audio = open(segment_name, "rb")
        transcripting = openai.audio.transcriptions.create(model="whisper-1", file=audio).text
        transcripts.append(transcripting)
        os.remove(segment_name)
    transcript = "\n".join(transcripts)

    return transcript

#passing transcript or each chucks to chatgpt
def generate_response(transcript):

    prompt = f"Translate {transcript} to english"

    completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You're a proffesional language translator"},
            {"role": "user", "content": prompt}
    ]
    )
    bot_first_response = completion.choices[0].message.content

    return bot_first_response

def audio_output_TTS(bot_response):
        outputs = syn.tts(bot_response)
        syn.save_wav(outputs, "audio-tts.wav")


def translate(youtube_link):

    # Use pytube to download the YouTube video
    yt = YouTube(youtube_link)
    stream = yt.streams.get_highest_resolution()
    file = stream.download(output_path='static', filename='my_video.mp4')
    filepath = os.path.join('static', 'my_video.mp4')

    # Transcribe video and generate timestamped transcript
    transcript = transcribe_video(filepath)
    print("Transcript youtubevid =")
    print(transcript)

    #opeanAI for the chat converation:
    nltk.download('punkt')

    if len(word_tokenize(transcript)) <= 3000:

        print("Token count less = ", len(word_tokenize(str(transcript))))
        bot_response = generate_response(transcript)
        print(f"less than 3000 tokens = {bot_response}\n")

    else:

        print("Token count more = ", len(word_tokenize(transcript)))
        chunk_size = 3000
        chunks = []
        sentences = sent_tokenize(transcript)
        current_chunk = ""

        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)

            if len(current_chunk.split()) + len(tokens) <= chunk_size:
                current_chunk += " " + sentence

            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

            print(f"TOKEN LENT OF unsent CHUNK = \n\n{len(word_tokenize(str(current_chunk.strip())))}\n\n\n")

        if current_chunk:
            chunks.append(current_chunk.strip())


        responses = []
        for chunk in chunks:
            response = generate_response(chunk)

            print(f"TOKEN LENT OF CHUNK = \n\n{len(word_tokenize(str(response)))}\n\n\n")

            responses.append(response)

        joined_response = ' '.join(responses)
        bot_response = joined_response


    audio_output_TTS(bot_response)
    
if __name__== "__main__":
    translate(str(sys.argv[1]))
