import speech_recognition as sr

IBM_USERNAME = "a8059080-91ae-4a3f-b5b7-8361ecdb52e5"
IBM_PASSWORD = "G0Fa60xnrWUE"


def speech_2_text(file_name):
    """
    :param file_name: Path of audio file
    :return: Text from speech
    """
    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(file_name) as source:
        # r.adjust_for_ambient_noise(source)
        r.dynamic_energy_threshold = True
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Sphinx
    try:
        # print("IBM watson : " + r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD))
        # print("Sphinx:	" + r.recognize_sphinx(audio))
        # print("Google:	" + r.recognize_google(audio))
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "#&*^&&^$^&#%@^#@()&(!!!!"
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))
