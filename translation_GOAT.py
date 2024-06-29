from flask import Flask, render_template, request, send_file, Response
from flask_bootstrap import Bootstrap
from pytube import YouTube
import openai

#Using socketIO to for js interaction:
from flask_socketio import SocketIO, emit
from flask import session
import os
import io

#Elevenlabs:
import requests

# For TTS (text-to-speach)
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import wave
import re

#chucking video:
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
import math

#for playing vidoe:
import uuid
from moviepy.audio.io.AudioFileClip import AudioFileClip

#chucking words of over 3000 tokens:
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Use your own API key
with open("openai-api-key.txt") as f:
	openai.api_key = f.read().strip()	

#Elevenlabs API key
with open("elevenlabs_api_key.txt") as f:
	user = f.read().strip()

transcript = []

conversation_history = []

bot_response = None

prompt = None

filepath = None

current_filepath = None

#voice = None
voice = "21m00Tcm4TlvDq8ikWAM" # elevenlabs voice-id

two_people = False

############# setup and load TTS-model #####################
with open("tts-models-path.txt") as f:
	path = f.read().strip()	#path = "/path/to/pip/site-packages/TTS/.models.json"
model_manager = ModelManager(path)
model_path, config_path, model_item = \
    model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC")
voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])
syn = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=voc_path,
    vocoder_config=voc_config_path
)
voc_path2, voc_config_path2, _ = model_manager.download_model("vocoder_models/en/ljspeech/multiband-melgan")
syn2 = Synthesizer(tts_checkpoint=model_path,
    tts_config_path=config_path,
    vocoder_checkpoint=voc_path2,
    vocoder_config=voc_config_path2)
############################################################


app = Flask(__name__)
app.config['SECRET_KEY'] = 'divine'
app.config['UPLOAD_FOLDER'] = 'static'

socketio = SocketIO(app)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('inputpage.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')


# Upload video page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global transcript
    global prompt
    global bot_response
    global conversation_history
    global filepath
    global current_filepath
    global two_people


    if request.method == 'POST':  
        checked = 'check' in request.form
        print(checked)      
        if request.form.get('two-people-checkbox'):
            two_people = True 
        else:
            two_people = False   
        two_people = 'two-people-checkbox' in request.form
        print("two_people = ", two_people)
        if 'file' in request.files:
            file = request.files['file']
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Transcribe video and generate timestamped transcript
            transcript = transcribe_video(filepath)
            print(transcript)
            current_filepath = filepath
            return render_template('audio.html', video_url=filepath, transcript=transcript)


        # Check if a YouTube link was provided
        elif 'youtube_link' in request.form:
            youtube_link = request.form['youtube_link']
            # Use pytube to download the YouTube video

            yt = YouTube(youtube_link)
            stream = yt.streams.get_highest_resolution()
            file = stream.download(output_path='static', filename='my_video.mp4')
            filepath = os.path.join('static', 'my_video.mp4')

            # Transcribe video and generate timestamped transcript
            transcript = transcribe_video(filepath)
            print("Transcript youtubevid =")
            print(transcript)
            current_filepath = filepath
            return render_template('audio.html', video_url=filepath, transcript=transcript)


        elif 'audio' in request.files:
            file = request.files['audio']
            filename = str(uuid.uuid4()) + '.' + file.filename.split('.')[-1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Transcribe video
            transcript = transcribe_audio(filepath)
            print(transcript)
            current_filepath = filepath
            return render_template('audio.html', filename=filepath, transcript=transcript)


        elif 'link' in request.form:
            link = request.form['link']
            response = requests.get(link)
            filename = str(uuid.uuid4()) + '.mp3'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(response.content)

            # Transcribe video and generate timestamped transcript
            transcript = transcribe_audio(filepath)
            print(transcript)
            current_filepath = filepath
            return render_template('audio.html', filename=filepath, transcript=transcript)

        return render_template('audio.html')
    else:

        return render_template('audio.html')


# Play video page
@app.route('/play/<path:video_url>')
def play(video_url):
    # Remove the extra 'static' directory from the file path
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], video_url.replace('static/', '', 1))
    return send_file(file_path, mimetype='video/mp4')


# Play audio on page
@app.route('/play_file/<path:filename>')
def play_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename.replace('static/', '', 1))
    return send_file(file_path, mimetype='audio/mp3')



#For generating the transcript with whisper
def transcribe_video(filepath):

    print("file =")
    print(filepath)
    video = VideoFileClip(filepath)
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


#For generating of audio the transcript with whisper
def transcribe_audio(filepath):

    audio = AudioFileClip(filepath)
    segment_duration = 10 * 60  # seconds
    transcripts = []
    num_segments = math.ceil(audio.duration / segment_duration)

    # Loop through the segments
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, audio.duration)
        segment = audio.subclip(start_time, end_time)
        segment_name = f"segment_{i+1}.mp3"
        segment.write_audiofile(segment_name)

        # Pass the audio segment to WISPR for speech recognition
        audio = open(segment_name, "rb")
        transcripting = openai.audio.transcriptions.create("whisper-1", audio).text
        transcripts.append(transcripting)
        os.remove(segment_name)
    transcript = "\n".join(transcripts)

    return transcript




#Training with the speakers voice
#old function name was the same and in conflict with: def transcribe_video(filepath):
def define_voice(filepath):
    global voice
    video = VideoFileClip(filepath)
    segment_duration = video.duration / 2
    num_segments = 2

    # Loop through the segments
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, video.duration)
        segment = video.subclip(start_time, end_time)
        segment_name = f"segment_{i+1}.mp3"
        segment.audio.write_audiofile(segment_name)
        
    #voice = get_audio("segment_1.mp3", "segment_2.mp3")
    voice = "21m00Tcm4TlvDq8ikWAM"

    # Delete the segment MP3 file
    os.remove("segment_1.mp3")
    os.remove("segment_2.mp3")
    print(f"Voice ID = {voice}")


#Training with the speakers voice
def get_audio(voice_i, voice_ii):

    add_voice_url = "https://api.elevenlabs.io/v1/voices/add"

    headers = {
        "Accept": "application/json",
        "xi-api-key": user #XI_API_KEY
    }

    data = {
        'name': 'Voice name',
        'labels': '{"accent": "American", "gender": "Female"}'
    }

    files = [
        ('files', ('sample1.mp3', open(voice_i, 'rb'), 'audio/mpeg')),
        ('files', ('sample2.mp3', open(voice_ii, 'rb'), 'audio/mpeg'))
    ]

    response = requests.post(add_voice_url, headers=headers, data=data, files=files)
    voice_id = response.json()[response.content]

    return voice_id




#opeanAI for the chat converation:
nltk.download('punkt')

@socketio.on('user_input')

def handle_conversation(user_input):

    print(f"Voice ID 2 = {voice}")

    global bot_response

    print("Transcript=")
    print(transcript)

    if len(word_tokenize(transcript)) <= 3000:

        print("Token count less = ", len(word_tokenize(str(transcript))))
        print("Two people = ", two_people)
        bot_response = generate_response(transcript, user_input, two_people)
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
            response = generate_response(chunk, user_input)

            print(f"TOKEN LENT OF CHUNK = \n\n{len(word_tokenize(str(response)))}\n\n\n")

            responses.append(response)

        joined_response = ' '.join(responses)
        bot_response = joined_response

    use_TTS = True
    if use_TTS:
        audio_output_TTS(bot_response, two_people)
        new_audio = wave.open("audio-tts.wav") # TODO: convert to mp3 or find a way to emit the wav?
    else:
        new_audio = audio_output_elevenlabs(bot_response, voice)

    # Create a Flask response object with the mp3 data and appropriate headers
    response = Response(new_audio, mimetype='audio/mpeg')
    response.headers.set('Content-Disposition', 'attachment', filename='responding.mp3')

    # Emit the audio data to the client-side
    socketio.emit('new_audio', {'data': new_audio, 'type': 'audio/mpeg'})
    socketio.emit('bot_response', bot_response)


#passing transcript or each chucks to chatgpt
def generate_response(transcript, user_input, two_people):

    if two_people:
        prompt = (f"Translate {transcript} to {user_input}. Indicate the two persons speaking "
            "by adding the keywords _speaker1_ and _speaker2_ and whitespaces on each "
            "side of the keyword. Do this each time the person starts to speak.")
    else:
        prompt = f"Translate {transcript} to {user_input}"

    #completion = openai.ChatCompletion.create(
    completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You're a proffesional language translator"},
            {"role": "user", "content": prompt}
    ]
    )
    bot_first_response = completion.choices[0].message.content

    return bot_first_response


#Eleven-labs: Text to audio for new lang
def audio_output_elevenlabs(bot_response, voice):

    print(voice)

    CHUNK_SIZE = 1024


    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": user
    }

    data = {
        "text": bot_response,
        "voice_settings": {
        "stability": 0,
        "similarity_boost": 0
        }
    }
    response = requests.post(url, json=data, headers=headers, stream=True)
    print(response)


    audio_data = io.BytesIO()
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            audio_data.write(chunk)

    return audio_data.getvalue()


def split_conversation(l, s):
    # Create a regex pattern that matches any of the speakers
    pattern = "|".join(l)

    # Use the pattern to split the string
    m = re.split(pattern, s)
    
    return [i.strip() for i in m if i] # removes leading/trailing white spaces and empty strings

def audio_output_TTS(bot_response, two_people):
    if two_people:
        print("Using two voices")
        # divide bot_response into list of strings containing speaker1, speaker2, speaker1 ...

        #infiles = ["sound_1.wav", "sound_2.wav"]
        outfile = "audio-tts-2p.wav"

        lines = split_conversation(["_speaker1_", "_speaker2_"], bot_response)

        data= []
        for line in lines:
            #w = wave.open(infile, 'rb')
            w = syn.tts(line)
            data.append( [w.getparams(), w.readframes(w.getnframes())] )
            w.close()
            
        output = wave.open(outfile, 'wb')
        output.setparams(data[0][0])
        for i in range(len(data)):
            output.writeframes(data[i][1])
        output.close()
    else:
        outputs = syn.tts(bot_response)
        syn.save_wav(outputs, "audio-tts.wav")
    
#Automatic delele
@app.route('/delete_video', methods=['POST'])
def delete_video():
    global current_filepath

    # Check if a video file path has been set
    if os.path.exists(current_filepath):
        os.remove(current_filepath)
        print("Dead & Gone")
    return "Ooops! Time out"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)