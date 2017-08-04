import pyaudio
import wave
from threading import Thread

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 3
WAVE_OUTPUT_FILENAME = "record-"
WAVE_OUTPUT_FOLDER = "./records/"


def save_file(frame, p, count, fn):
    p = pyaudio.PyAudio()
    name = WAVE_OUTPUT_FOLDER + WAVE_OUTPUT_FILENAME + str(count) + ".wav"
    wf = wave.open(name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frame))
    wf.close()
    # print "File saved in: " + name
    fn[0] = name


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None

    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(self._Thread__args,
                                                **self._Thread__kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def record(file_name):
    p = pyaudio.PyAudio()
    print("Start Record!!!")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    count = 0
    while True:
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        # file_name[0] = save_file(frames, p, count)
        savefile = Thread(target=save_file, args=(frames, p, count, file_name,))
        savefile.start()
        count = count + 1
    stream.stop_stream()
    stream.close()
    p.terminate()
"""a = [""]
record(a)"""