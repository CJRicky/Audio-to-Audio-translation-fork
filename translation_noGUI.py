from pytube import YouTube
import yt_dlp # replacing pytube

import os
from glob import glob

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
import torch
pipeline.to(torch.device("cuda"))


# Use your own API key
with open("openai-api-key.txt") as f:
	openai.api_key = f.read().strip()	

############# setup and load TTS-model #####################
syn, syn2 = load_tts_model()
############################################################
def run_whisper(audio):
    segment_duration = 30  # seconds
    print("Audio duration = ", audio.duration)
    transcripts = []
    num_segments = math.ceil(audio.duration / segment_duration)
    print("Number of segments =", num_segments)

    # Loop through the segments
    for i in range(num_segments):
        # Calculate the start and end times for the current segment
        print("Transcribing segment nr: ", i+1)
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, audio.duration)
        segment = audio.subclip(start_time, end_time)
        print("Audio-segment duration: ", segment.duration)
        segment_name = f"segment_{i+1}.mp3"
        segment.write_audiofile(segment_name)

        # Pass the audio segment to WISPR for speech recognition
        audio_segment = open(segment_name, "rb")
        transcripting = openai.audio.transcriptions.create(model="whisper-1", file=audio_segment).text
        transcripts.append(transcripting)
        os.remove(segment_name)
    transcript = "\n".join(transcripts)

    return transcript

#passing transcript or each chucks to chatgpt
def generate_response(transcript):

    #prompt = f"Translate {transcript} to english."
    prompt = f"Translate the following text to english: {transcript}"

    completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You're a proffesional language translator. In all sentances in the translation: Replace '...' with '.',\
              '. And' with ' and' and translate letters 'å', 'ä', and 'ö' to their English equivalents 'aa', 'ae', and 'oe'.\
              Just return the translation. No text before."},
            {"role": "user", "content": prompt}
    ]
    )
    bot_first_response = completion.choices[0].message.content

    bot_first_response.replace("...",".")
    bot_first_response.replace(". And",", and")
    bot_first_response.replace(". and",", and")

    return bot_first_response


def get_translation(transcript): 
    #opeanAI for the chat converation:
    nltk.download('punkt')   # Move to run just once? 

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

        bot_response = ' '.join(responses)
    return  bot_response

def audio_output_TTS(bot_response, filename):
        outputs = syn.tts(bot_response)
        syn.save_wav(outputs, filename)


#For generating the transcript with whisper
def transcribe_video(filepath):

    print("file =")
    print(filepath)
    video = VideoFileClip(filepath)
    audio = video.audio

    mp3_file = "./mp3-files/audio.mp3"
    wav_file = "./wav-files/audio.wav"
    audio.write_audiofile(mp3_file)

    audio_wav = AudioSegment.from_mp3(mp3_file)
    audio_wav.export(wav_file, format="wav")
    #audio = wave.open(wav_file, mode='rb')

    # apply pretrained pipeline
    print("Finding speakers (start and stop time for each speaker)...")
    diarization = pipeline(wav_file)

    speakers = []
    start = []
    stop = []
    previous_speaker = ""
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        if speaker == previous_speaker:
             stop[-1] = turn.end
             continue
        else:
            speakers.append(speaker)
            previous_speaker = speaker
            start.append(turn.start)
            stop.append(turn.end)

    print(speakers)
    start[0] = 0
    print(start)
    print(stop)
    #start[1:-1] = stop[0:-2]
    #print(speakers)
    #print(start)
    #print(stop)

    print("Transcribing using Whisper...")
    N_speakers = len(speakers)
    for idx, speaker in enumerate(speakers):
        start_idx = round(start[idx]*1e3) 
        end_idx = round(stop[idx]*1e3) 
        start_time = start[idx]
        end_time = stop[idx]
        if idx == 0:
             start_idx = 0
             start_time = 0
        elif idx == N_speakers-1: 
            end_idx = round(audio.duration)
            end_time = audio.duration
        #chunk = audio_wav[start_idx:end_idx]
        print(audio.duration)
        chunk = audio.subclip(start_time, end_time) # what time format, s, ms??
        #chunk_name = f"chunk_{idx+1}.mp3"
        #chunk.write_audiofile(chunk_name)

        # Pass the audio segment to WISPR for speech recognition
        #audio_chunk = open(chunk_name, "rb")
        transcript = run_whisper(chunk)   
        print(transcript)
        
        # Translate
        if len(word_tokenize(str(transcript))) > 0:
               translation = get_translation(transcript)
               print(translation)
        else:
             continue # nothing to translate
        
        # Text to audio. select speaker voice
        filename = f"./wav-files/audio-tts-{idx+1}.wav"
        print("Writing translation to: ", filename)
        audio_output_TTS(translation, filename)

        print("File " + filename + " done!")

    # join all the wav-files into one
    wav_files = glob('./wav-files/audio-tts-*.wav')
    print("wav-files: ")
    print(wav_files[0])

    numbers = []
    for wf in wav_files:
        result = re.search('tts-(.*).wav', wf)
        numbers.append(int(result.group(1)))

    sortIdx = [i[0] for i in sorted(enumerate(numbers), key=lambda x:x[1])]

    wav_files = [wav_files[i] for i in sortIdx]

    infiles = wav_files
    outfile = "./wav-files/audio-translated.wav"
    data= []
    for infile in infiles:
        w = wave.open(infile, 'rb')
        data.append( [w.getparams(), w.readframes(w.getnframes())] )
        w.close()
        
    output = wave.open(outfile, 'wb')
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()

    # TODO: clear tmp-files.
    # TODO: use different voices

    input("press any key to contrinue")



def translate(youtube_link):

    # Use pytube to download the YouTube video
    # TODO: only download audio?
    use_pytube = False
    filename = 'my_video.mp4'
    if use_pytube:
        yt = YouTube(youtube_link)
        stream = yt.streams.get_highest_resolution()
        file = stream.download(output_path='./youtube-dl', filename=filename)
    else:      
        yt_opts = {'outtmpl': filename}

        ydl = yt_dlp.YoutubeDL(yt_opts)
        
        ydl.download(youtube_link)

    filepath = os.path.join('static', 'my_video.mp4')

    # Transcribe video and generate timestamped transcript
    transcribe_video(filepath)
    
if __name__== "__main__":
    if __debug__:
        print('Debug ON')
        link = 'https://www.youtube.com/watch?v=z-Rpmb9wxK0'
    else:
        print('Debug OFF')
        link = str(sys.argv[1])
        link = 'https://www.youtube.com/watch?v=z-Rpmb9wxK0'
        
    translate(youtube_link=link)
