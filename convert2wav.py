# ffmpeg -i uno.mp3 -acodec pcm_s16le -ar 44100  uno.wav


from shellfish import sh
from subprocess import run
from shlex import quote
mp3_filepaths = (f for f in sh.files_gen("the-daily" ) if f.endswith('.mp3'))

def no_non_ascii(s):
    return ''.join([c for c in s if ord(c) < 128])
for f in mp3_filepaths:
    print(f)
    clean_name = f.replace('"', '').replace(' ', '-').replace('\'', '').replace('(', '').replace(')', '').replace('&', '').replace('!', '').replace(',', '').replace('?', '').replace(';', '').replace(':', '').replace('\"', '')
    # get rid of double quotes
    clean_name = no_non_ascii(clean_name)
    
    output_file = clean_name.replace(".mp3", ".wav").replace('the-daily', 'the-daily-wav')
    print(output_file)
    if sh.exists(output_file):
        print('exists')
        continue
    sh.copy_file(f, clean_name.replace('the-daily', 'the-daily-wav'))
    args = [
            'ffmpeg', '-i', quote(clean_name.replace('the-daily', 'the-daily-wav').replace('\"', '\\"').replace('\\', '/')), '-acodec', 'pcm_s16le', '-ar', '44100', quote(output_file.replace('\"', '\\"').replace('\\', '/'))
        ]
    print(args)
    proc = run(
        args
    )
    # break