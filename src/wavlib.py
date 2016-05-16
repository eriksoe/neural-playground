# Library for reading/writing (some) WAV files.

import wave
import struct

def read_wav(filename):
    w = wave.open(filename, 'rb')
    try:
        channels = w.getnchannels()
        swidth = w.getsampwidth()
        fcount = w.getnframes()
        frames = w.readframes(fcount)
    finally:
        w.close()

    frame_size = channels * swidth

    if (channels==2 and swidth == 2):
        unpacker = frame_unpacker_16bit_stereo
    else:
        raise Exception("Unhandled WAV flavour.")

    samples = []
    for i in range(fcount):
        samples.append(unpacker(frames, i*frame_size))
    return samples

def frame_unpacker_16bit_stereo(data, offset):
    (s1,s2) = struct.unpack_from("<hh", data, offset)
    return (float(s1+s2)) / (1<<16) # Scale to [-1;1]


def write_wav(filename, samples):
    data = ''
    for sample in samples:
        sample = max(-1.0, min(1.0, sample)) # Clamp.
        frame = struct.pack('<h', int(sample * (1<<15) - 0.5) )
        data += frame

    w = wave.open(filename, 'wb')
    try:
        w.setnchannels(1) # Mono
        w.setsampwidth(2) # 16 bit
        w.setframerate(48000) # Fixed, for now.
        w.setnframes(len(samples))
        w.writeframes(data)
    finally:
        w.close()
