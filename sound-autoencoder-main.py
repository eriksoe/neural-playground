#!/usr/bin/python

# Sound Auto-encoder, main program.

import os
import random

import wavlib
import distortlib
import spectrumlib
import neurallib
import numpy

#========== Configuration ========================================
training_data_path = "data/training"
validation_data_path = "data/validation"
distort_prob = 0.3

#========== Utility functions ====================
def load_sounds_from_dir(dir_path):
    sounds = []
    for filename0 in os.listdir(dir_path):
        if os.path.splitext(filename0)[1] == '.wav':
            filename = os.path.join(dir_path,filename0)
            if os.path.isfile(filename):
                print "Loading '%s'..." % (filename,)
                sounds.append(wavlib.read_wav(filename))
    return sounds

def pad_sound(sound, seg_size):
    "Pad sound at end so that its length is a multiple of seg_size."
    pad_size = (seg_size-1) - (len(sound)-1) % seg_size
    return sound + ([0] * pad_size)

#========== Main program state ====================

class SoundAutoencoderProgram:
    seg_size = 1024

    def __init__(self, training_data_path, validation_data_path):
        self.training_data = load_sounds_from_dir(training_data_path)
        self.validation_data = load_sounds_from_dir(validation_data_path)
        print "Loaded: %d training sounds" % (len(self.training_data),)
        print "Loaded: %d validation sounds" % (len(self.validation_data),)

        self.net = neurallib.NeuralNetwork((3,100), (100,), [100])
        self.trainer = neurallib.MiniBatchTrainer(self.net, 10)

    def pick_training_example(self):
        org_sound = random.choice(self.training_data)
        sound = org_sound
        for distortion in distortlib.distortion_collection:
            if random.uniform(0,1) < distort_prob:
                amount = random.uniform(0,1)
                before = len(sound)
                sound = distortion(sound, amount)
                after = len(sound)
                if before != after:
                    print "BAD distortion: %s" % distortion

        return (org_sound,sound)

    def prepare_sample(self, sound, ref_sound):
        seg_size = self.seg_size

        length = len(sound)
        offset = int(random.uniform(0, seg_size))
        sound = pad_sound(sound[offset:], seg_size)
        ref_sound = pad_sound(ref_sound[offset:], seg_size)

        ampls = spectrumlib.enwindow(sound, seg_size)
        freqs = spectrumlib.forward_fft(ampls)

        ref_ampls = spectrumlib.enwindow(ref_sound, seg_size)
        ref_freqs = spectrumlib.forward_fft(ref_ampls)

        #print "%d/%d ; %d/%d" % (len(sound), len(ref_sound), len(freqs), len(ref_freqs))
        samples = []
        for i in range(len(freqs)):
            this_seg = freqs[i][0:100]
            last_seg = freqs[i-1][0:100] if i>0 else numpy.zeros((100,))
            next_seg = freqs[i+1][0:100] if i+1<len(freqs) else numpy.zeros((100,))
            inputs = numpy.array([last_seg,this_seg,next_seg], dtype='float')
            outputs = ref_freqs[i][0:100]
            samples.append( (inputs, outputs) )
        return samples

    def train_one_sound(self):
        (ref_sound,sound) = self.pick_training_example()
        training_sets = self.prepare_sample(sound, ref_sound)
        # TODO: make and use a NeuralNetwork trainer which handles minibatching.
        for (inputs, outputs) in training_sets:
            self.trainer.present_one_example(inputs, outputs)

prg = SoundAutoencoderProgram(training_data_path, validation_data_path)
#(org_sound,sound) = prg.pick_training_example()
#wavlib.write_wav("dump.wav", sound)
#wavlib.write_wav("dump2.wav", org_sound)
for i in range(100):
    prg.train_one_sound()
