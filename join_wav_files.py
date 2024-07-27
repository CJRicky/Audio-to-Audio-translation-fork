from pydub import AudioSegment
from glob import glob
import re
import wave

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
