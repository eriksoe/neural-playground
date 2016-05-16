# This library contains the building blocks for performing the transformation
#    Time domain -> windowing -> FFT -> Frequency domain
# and vice versa.

import math
import numpy as np
import numpy.random as rnd
from numpy.fft import fft

def enwindow(a, seg_size):
    half_size = seg_size//2
    window = np.hanning(seg_size)
    seg_cnt = len(a) // half_size

    ampls = np.empty((seg_cnt+1, seg_size))
    for seg_nr in range(seg_cnt+1):
        t0 = (seg_nr-1) * half_size
#        print "Time %s-%s:" % (t0, t0+seg_size-1)
        if t0<0:                    # The first:
            wseg = a[t0+half_size : t0+seg_size] * window[half_size:]
            wseg = np.concatenate([np.zeros((half_size,)), wseg])
        elif t0+half_size>=len(a): # The last:
            wseg = a[t0 : t0+half_size] * window[0:half_size]
            wseg = np.concatenate([wseg, np.zeros((half_size,))])
        else:
            wseg = a[t0 : t0+seg_size] * window
        ampls[seg_nr] = wseg
    return ampls

def dewindow(a):
    seg_cnt = a.shape[0]
    seg_size = a.shape[1]
    half_size = seg_size//2
    window = np.hanning(seg_size)
    sample_cnt = half_size * (seg_cnt - 1)

    samples1 = np.empty(sample_cnt)
    samples2 = np.empty(sample_cnt)
    samples = np.empty(sample_cnt)
#    print "seg_size = %s" % (seg_size,)
#    print "sample_cnt = %s" % (sample_cnt,)

    for seg_nr in range(seg_cnt-1):
        samples[seg_nr*half_size : (seg_nr+1)*half_size] = (
            a[seg_nr][half_size:] * window[half_size:]
            +
            a[seg_nr+1][0:half_size] * window[0:half_size])
    return samples

# forward_fft : ampl-windows -> freq-windows
def forward_fft(ws):
    seg_cnt = ws.shape[0]
    freq_cnt = ws.shape[1]//2 + 1
    print "freq_cnt = %s" % (freq_cnt,)
    freqs = np.empty((seg_cnt, freq_cnt), dtype=complex)
    for i in range(seg_cnt):
        freqs[i] = np.fft.rfft(ws[i])
        #freqs[i] = abs(freqs[i])
    return freqs

# backward_fft : freq-windows -> ampl-windows
def backward_fft(ws):
    seg_cnt = ws.shape[0]
    ampl_cnt = (ws.shape[1] - 1)*2
    print "ampl_cnt = %s" % (ampl_cnt,)
    ampls = np.empty((seg_cnt, ampl_cnt), dtype=float)
    for i in range(seg_cnt):
        ampls[i] = np.fft.irfft(ws[i]*(0.707+0.707j)).real
    return ampls

