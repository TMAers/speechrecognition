import wave
import time
import datetime
from pydub import AudioSegment
a=datetime.datetime.now()

infiles = ["3_123286-0028.wav", "5_134686-0012.wav"]
outfile = "output111.wav"
data= []
print len(data)
for infile in infiles:
    w = wave.open(infile, 'rb')
    data.append( [w.getparams(), w.readframes(w.getnframes())] )
    w.close()
print len(data)
output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
output.writeframes(data[0][1])
output.writeframes(data[1][1])
#output.writeframes(data[2][1])
output.close()

