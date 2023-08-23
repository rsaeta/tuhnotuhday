# import wave
# import numpy as np
# import matplotlib.pyplot as plt

# signal_wave = wave.open('uno.wav', 'r')
# sample_rate = 16000
# sig = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)
# sig = sig[:]
# sig = sig[25000:32000]
# left, right = data[0::2], data[1::2]
# plt.figure(1)

# plot_a = plt.subplot(211)
# plot_a.plot(sig)
# plot_a.set_xlabel('sample rate * time')
# plot_a.set_ylabel('energy')

# plot_b = plt.subplot(212)
# plot_b.specgram(sig, NFFT=1024, Fs=sample_rate, noverlap=900)
# plot_b.set_xlabel('Time')
# plot_b.set_ylabel('Frequency')

# # save plot to file
# plt.savefig('plot.png', bbox_inches='tight')
# plt.show()

def plot_shit():

    import matplotlib.pyplot as plt
    import numpy as np
    import wave

    file = 'uno.wav'

    wav_file = wave.open(file,'r')

    #Extract Raw Audio from Wav File
    signal = wav_file.readframes(-1)
    if wav_file.getsampwidth() == 1:
        signal = np.array(np.frombuffer(signal, dtype=np.uint8)-128, dtype=np.uint8)
    elif wav_file.getsampwidth() == 2:
        signal = np.frombuffer(signal, dtype=np.int16)
    else:
        raise RuntimeError("Unsupported sample width")

    # http://schlameel.com/2017/06/09/interleaving-and-de-interleaving-data-with-python/
    deinterleaved = [signal[idx::wav_file.getnchannels()] for idx in range(wav_file.getnchannels())]

    print(signal.shape, )
    print({
        "max": np.nanmax(signal), 
        "min": np.nanmin(signal), 
        "mean": np.nanmean(signal), 
        "std": np.nanstd(signal), 
        "median": np.nanmedian(signal)}
        )
    #Get time from indices
    fs = wav_file.getframerate()
    Time=np.linspace(0, int(len(signal)/wav_file.getnchannels()/fs), num=int(len(signal)/wav_file.getnchannels()))
    plt.figure(figsize=(200,3))
    #Plot
    plt.figure(1)
    #don't care for title
    #plt.title('Signal Wave...')
    for channel in deinterleaved:
        plt.plot(Time,channel, linewidth=.125)
    #don't need to show, just save
    #plt.show()
    plt.savefig('uno.png', dpi=72)
    # ffmpeg -i uno.mp3 -acodec pcm_s16le -ar 44100  uno.wav


import wave
import audioop
from os import path, makedirs
from shutil import rmtree

leading_frames = 100
rmtree('chunks', ignore_errors=True)
makedirs('chunks', exist_ok=True)
# Open the WAV file
with wave.open('uno.wav', 'rb') as wav_file:
    # Get the sample rate and number of channels
    sample_rate = wav_file.getframerate()
    num_channels = wav_file.getnchannels()
    # Read all the frames from the WAV file
    frames = wav_file.readframes(wav_file.getnframes())
    # Calculate the threshold for detecting pauses (you can adjust this as needed)
    threshold = 8000
    # Initialize variables for tracking the start and end of each chunk
    chunk_start = 0
    chunk_end = 0
    # Initialize a list to hold the chunk data
    chunks = []
    # Loop through all the frames
    for i in range(0, len(frames), sample_rate):
        # Calculate the energy level of the current chunk of audio
        chunk_energy = audioop.rms(frames[i:i+sample_rate], 2)
        # Check if the energy level is below the threshold (i.e., there is a pause)
        if chunk_energy < threshold:
            # If this is the first pause, set the chunk start time
            if chunk_start == 0:
                chunk_start = i / sample_rate
            # Otherwise, set the chunk end time and add the chunk data to the list
            else:
                chunk_end = i / sample_rate
                # Create a dictionary to store the chunk data and metadata
                chunk_start_int = int(chunk_start * sample_rate) 
                if chunk_start_int > leading_frames:
                    chunk_start_int -= leading_frames
                chunk_stop_int = int(chunk_end * sample_rate)
                if chunk_stop_int < len(frames) - leading_frames:
                    chunk_stop_int += leading_frames
                chunk = {
                    'start_time': chunk_start,
                    'end_time': chunk_end,
                    # 'data': frames[chunk_start*sample_rate:chunk_end*sample_rate]
                    'data': frames[chunk_start_int:chunk_stop_int]
                }
                chunks.append(chunk)
                # Reset the variables for the next chunk
                chunk_start = 0
                chunk_end = 0
        # If there is no pause, just continue to the next chunk
        else:
            continue
    # If there is a remaining chunk at the end of the audio, add it to the list
    if chunk_start != 0:
        # Create a dictionary to store the chunk data and metadata
        chunk = {
            'start_time': chunk_start,
            'end_time': wav_file.getnframes() / sample_rate,
            'data': frames[chunk_start*sample_rate:]
        }
        chunks.append(chunk)


    shapes = [len(chunk['data']) for chunk in chunks]
    print('len frames', len(frames))
  # Write each chunk to a new WAV file
    for i, chunk in enumerate(chunks):
        # Create a new WAV file with the same parameters as the original file
        output_file = wave.open(f'chunks\\chunk_{i}.wav', 'wb')
        output_file.setnchannels(num_channels)
        output_file.setsampwidth(wav_file.getsampwidth())
        output_file.setframerate(sample_rate)
        # Write the chunk data to the new file
        output_file.writeframes(chunk['data'])
        output_file.close()

# print(shapes)