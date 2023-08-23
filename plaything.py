filepath = 'chunks\\chunk_83.wav'
# import playsound
# playsound.playsound(filepath)

import winsound
winsound.PlaySound(
    filepath
    , winsound.SND_FILENAME)
