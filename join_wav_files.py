from pydub import AudioSegment
from glob import glob
import re

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

"""for wf_idx, wf in enumerate(wav_files):
    print(wf_idx)
    sound_tmp = AudioSegment.from_wav(wf)
    if wf_idx == 0:
            print("hej")
            print(wf)
            combined_sounds = sound_tmp
    else:
            print("tja")
            print(wf)
            combined_sounds + sound_tmp

combined_sounds.export("./wav-files/audio-translated.wav", format="wav")"""


import wave

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
