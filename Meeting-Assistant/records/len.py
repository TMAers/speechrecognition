from pydub import AudioSegment

# t1 = t1 * 1000 #Works in milliseconds
# t2 = t2 * 1000
newAudio = AudioSegment.from_wav("record-0.wav")
# newAudio = newAudio[t1:t2]
# newAudio.export('newSong.wav', format="wav")
"""for i in range(0,4):
    t1=i*2000
    t2=(i+1)*2000
    newaudio=newAudio[t1:t2]
    newaudio.export('3_123286-0028{0}.wav'.format(i), format="wav")"""
print len(newAudio)

