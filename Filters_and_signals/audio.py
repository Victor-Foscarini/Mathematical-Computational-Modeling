#Class that contains generations and filtering of signals
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pydub import AudioSegment
from pydub.playback import play
from scipy.io import wavfile
from scipy import signal

class Audio:

    #Signal rate and duration
    rate = 44100 # determine number of data points the signal use to represnt wave by second
    duration = 4 # general duration of audios (may or may not be)

    def __init__(self):
        self.tone = []
        self.freq = 0

    def generate_wave(self,freq,func,duration=duration,rate=rate,save=True):
        t = np.linspace(0,duration,rate*duration) #time elapsed
        f = func(freq*t)
        if save==True :
            self.atualize_audio(f)
        self.atualize_freq(freq,duration)
        return f

    #Generate signal with cosine wave of frequency
    def generate_cosine_wave(self,freq,amp=1,f0=0,duration=duration,rate=rate,save=True):
        t = np.linspace(0,duration,rate*duration) #time elapsed
        f = amp*np.cos(2*np.pi*freq*t+f0) #consertar f0
        if save==True :
            self.atualize_audio(f)
        self.atualize_freq(freq,duration)
        return f

    def generate_cosine_exp_wave(self,freq,amp=1,factor=-1,f0=0,duration=duration,rate=rate,save=True):
        t = np.linspace(0,duration,rate*duration) #time elapsed
        f = amp*np.exp(factor*t)*np.cos(2*np.pi*freq*t+f0)
        if save==True :
            self.atualize_audio(f)
        self.atualize_freq(freq,duration)
        return f

    def atualize_audio(self,audio):
        self.tone = np.concatenate((self.tone,audio))

    def scale_audio(self,factor):
        self.tone *= factor

    def clean_audio(self):
        self.tone = []

    def atualize_freq(self,freq,duration=duration): #smallest frequency dominate
        if self.freq==0 or freq<self.freq :
            self.freq = freq

    def plot_signal(self,title="Signal",fontsize=20,Nseconds=1.0,figsize=(16,3),ylim=True):
        plt.figure(figsize=figsize)
        end = int(Nseconds*self.rate) #show only first Nseconds
        time = np.linspace(0,Nseconds,int(self.rate*Nseconds))
        plt.plot(time[:end],self.tone[:end])
        plt.title(title,fontsize=fontsize)
        if ylim:
            plt.ylim((-1,1))
        plt.show()

    def mix_signals(self,signals):
        if self.tone == []:
            self.tone = [0]*len(signals[0].tone)
        for signal in signals:
            self.tone += signal.tone
            self.atualize_freq(signal.freq,signal.duration)

    def save_audio(self,filename):
        normalized_tone = np.int16((self.tone / self.tone.max()) * 32767) #normalize for storage
        write(filename,self.rate,normalized_tone)

    def read_audio(self,filename):
        audio = AudioSegment.from_wav(filename)
        return audio

    def play_audio(self,filename):
        audio = AudioSegment.from_wav(filename)
        play(audio)

    def get_audio(self,filename):
        fs, x = wavfile.read(filename)
        self.tone = x

    def apply_high_pass(self,filename,cutoff=1000,resize=True,ordem=101):
        fs, x = wavfile.read(filename)      # 16-bit mono 44.1 khz
        b = signal.firwin(ordem,cutoff=cutoff,fs=fs,pass_zero='highpass')
        x = signal.lfilter(b, [1.0], x)
        wavfile.write(filename, fs, x.astype(np.int16))
        self.tone = x
        if resize:
            self.tone = self.tone/32767

    def apply_low_pass(self,filename,cutoff=1000,resize=True,ordem=101):
        fs, x = wavfile.read(filename)      # 16-bit mono 44.1 khz
        b = signal.firwin(ordem,cutoff=cutoff,fs=fs,pass_zero='lowpass')
        x = signal.lfilter(b, [1.0], x)
        wavfile.write(filename, fs, x.astype(np.int16))
        self.tone = x
        if resize:
            self.tone = self.tone/32767

    def apply_band_pass(self,filename,cutoff=[900,1000],resize=True,ordem=101):
        fs, x = wavfile.read(filename)      # 16-bit mono 44.1 khz
        b = signal.firwin(ordem,cutoff=cutoff,fs=fs,pass_zero='bandpass')
        x = signal.lfilter(b, [1.0], x)
        wavfile.write(filename, fs, x.astype(np.int16))
        self.tone = x
        if resize:
            self.tone = self.tone/32767
