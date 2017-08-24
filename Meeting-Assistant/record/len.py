from pydub import AudioSegment

# t1 = t1 * 1000 #Works in milliseconds
# t2 = t2 * 1000
newAudio = AudioSegment.from_wav("record-0.wav")
print len(newAudio)

