import math

import librosa
import loristrck as lt
import numpy as np
import pd
from scipy.signal.windows import blackman


class PeakList:
    def __init__(self):
        self.peaks = []
        self.partials = []
        self.sr = None
        self.total_samples = 0

    def addPeak(self, peak):
        self.peaks.append(peak)


    def getPeak(self):
        return self.peaks


    def getSr(self):
        return self.sr
    
    def setSr(self, sr):
        self.sr = sr

    def setTotalSamples(self, total_samples):
        self.total_samples = total_samples

    def getTotalSamples(self):
        return self.total_samples

    def setPartials(self, partials):
        if self.sr is None:
            self.sr = partials[0].sr
        self.partials.append(partials)


    def getPartials(self) -> list:
        return self.partials

# Peak Class
class Peak:
    def __init__(self, sr, N, sample_onset):
        self.onset = sample_onset * (1/ sr)
        self.index = -1
        self.freq = 0 
        self.amp = -1000
        self.phrase = None
        self.BackMatch = None
        self.ForwardMatch = None

        # Extra
        self.sr = sr
        self.N = N
        self.p = 0.0
        self.onsetSample = sample_onset

    def __repr__(self):
        back = "None" if self.BackMatch is None else self.BackMatch.index
        forward = "None" if self.ForwardMatch is None else self.ForwardMatch.index
        return "Peak(i={}, f={:.2f}, a={:.2f}, p={:.2f}, back={}, forward={})".format(self.index, self.freq, self.amp, self.phrase, back, forward)

    def getFreq(self, a, b, c, k):
        self.p = 1 / 2 * ((a - c) / (a - (2 * b) - c))
        self.freq = (self.p + k) * (self.sr / self.N)

    def getAmp(self, a, b, c):
        new_a = b.real - (1 / 4) * self.p * (a.real - c.real)
        new_b = b.imag - (1 / 4) * self.p * (a.imag - c.imag)
        self.amp = 20 * math.log10(math.sqrt(new_a ** 2 + new_b ** 2) )

    def getPhrase(self, b):
        self.phrase = math.atan2(b.imag, b.real)

    def getOnset(self, onset):
        self.onset = onset

    def checkPeakIndex(self, index):
        if self.index == -1:
            self.index = index
            if self.ForwardMatch is not None:
                self.ForwardMatch.checkPeakIndex(index)

        matching_classes = []
        current_class = self
        while current_class is not None:
            if current_class.index == index:
                matching_classes.append(current_class)
            current_class = current_class.ForwardMatch
        return matching_classes

# =======================
def stft_tab(name):
    samples = pd.tabread(name)
    sr = pd.get_sample_rate()
    N = 4096
    hop_size = 1024
    window = blackman(N) # Calcula *\eqinline{x(n) = 0.42 - 0.5 \cdot \cos\left(\frac{2\pi n}{N}\right) + 0.08 \cdot \cos\left(4\pi n / N\right)}*
    Xk_List = librosa.stft(samples, n_fft=N, hop_length=hop_size, window=window, pad_mode="constant")
    # sums_stft = np.abs(sftf)

    # Normalize
    normalized_stft = np.multiply(Xk_List, N / hop_size) # Normalizar a amplitude

    Peaks = PeakList()
    len_time = 0
    for frame_i in range(normalized_stft.shape[1]):
        X = normalized_stft[:, frame_i]
        foto_Peaks = []
        for k in range(1, X.shape[0] - 1):
            a = math.sqrt(X[k - 1].real ** 2 + X[k - 1].imag ** 2)
            b = math.sqrt(X[k].real ** 2 + X[k].imag ** 2)
            c = math.sqrt(X[k + 1].real ** 2 + X[k + 1].imag ** 2)
            if b > a and b > c: # LOCAL MAXIMA
                time = frame_i * hop_size 
                if time > len_time:
                    len_time = time
                newPeak = Peak(sr, N, time)
                newPeak.getFreq(a, b, c, k)
                newPeak.getAmp(X[k - 1] / N, X[k] / N, X[k + 1] / N)
                newPeak.getPhrase(X[k])
                newPeak.getOnset(frame_i * hop_size * (1 / sr))
                if newPeak.amp < -95:
                    continue
                foto_Peaks.append(newPeak)
        Peaks.addPeak(foto_Peaks)
    Peaks.setTotalSamples(len_time)
    return Peaks

def spear_pt_live(Peaks: PeakList, freq_threshold=20):
    for i in range(1, len(Peaks.peaks)):
        for curpeak in Peaks.peaks[i]:
            for prev_peak in Peaks.peaks[i - 1]:
                distance = abs(curpeak.freq - prev_peak.freq)
                if distance < freq_threshold:
                    if curpeak.BackMatch is not None:
                        existing_distance = abs(curpeak.BackMatch.freq - curpeak.freq)
                    else:
                        existing_distance = freq_threshold
                    if distance < existing_distance:
                        if curpeak.BackMatch is not None:
                            curpeak.BackMatch.ForwardMatch = None
                        curpeak.BackMatch = prev_peak
                        prev_peak.ForwardMatch = curpeak

    i_peaks = []
    peaks = Peaks.peaks

    # Valor em samples do último onset
    max_onset_sample = max(p.onsetSample for sublist in peaks for p in sublist)
    
    # Lista de listas seguindo o index. Portanto, 
    # i_peaks[0] serão todos os parciais que aparecem na primeira foto
    # i_peaks[1] na segunda foto, e assim sucessivamente.
    for _ in range(max_onset_sample // 1024 + 1):
        i_peaks.append([])
    
    # Organiza
    for sublist in peaks:
        for p in sublist:
            category_index = p.onsetSample // 1024
            i_peaks[category_index].append(p)

    return i_peaks 


def spear_pt(Peaks: PeakList, freq_threshold=20):
    totaldeSamples = Peaks.getTotalSamples()
    for i in range(1, len(Peaks.peaks)):
        for curpeak in Peaks.peaks[i]:
            for prev_peak in Peaks.peaks[i - 1]:
                distance = abs(curpeak.freq - prev_peak.freq)
                if distance < freq_threshold:
                    if curpeak.BackMatch is not None:
                        existing_distance = abs(curpeak.BackMatch.freq - curpeak.freq)
                    else:
                        existing_distance = freq_threshold
                    if distance < existing_distance:
                        if curpeak.BackMatch is not None:
                            curpeak.BackMatch.ForwardMatch = None
                        curpeak.BackMatch = prev_peak
                        prev_peak.ForwardMatch = curpeak
    index = 0
    partialTracking = PeakList()
    for peaks in Peaks.peaks:
        for peak in peaks:
            if peak.index != -1:
                continue
            peakList = peak.checkPeakIndex(index)
            partialTracking.setPartials(peakList)
            index += 1
    partialTracking.setTotalSamples(totaldeSamples)
    pd.print("Partial Tracking Done!")
    return partialTracking


def synth(peaksList: PeakList):
    samples = np.zeros(peaksList.getTotalSamples())
    for peaks in peaksList.getPartials():
        for peak_i in range(1, len(peaks)):
            phrases = peaks[peak_i - 1].phrase
            amplitude = np.linspace(peaks[peak_i - 1].amp, peaks[peak_i].amp, 1024)
            linAmp = 10 ** (amplitude / 20)
            freq = np.linspace(peaks[peak_i - 1].freq, peaks[peak_i].freq, 1024)
            signal = linAmp * np.sin(2 * np.pi * freq * np.arange(1024) / peaks[peak_i].sr + phrases)
            signalOnset = peaks[peak_i].onsetSample
            if (signalOnset + 1024) < len(samples):
                samples[signalOnset:signalOnset + 1024] += signal
    pd.tabwrite("pt", samples, resize=True)
    pd.print("Synth Done!")


def speat_livesynth(peaksList):
    if peaksList is None:
        return np.zeros(1024)
    total = len(peaksList)
    index = pd.get_obj_var("index", initial_value=0)
    if index >= total:
        return np.zeros(1024)
    sr = pd.get_sample_rate()
    signal = np.zeros(pd.get_vec_size())
    for peak in peaksList[index]:
        this = peak
        next = peak.ForwardMatch
        if next is None:
            continue
        phrases = this.phrase
        amplitude = np.linspace(this.amp, next.amp, 1024)
        linAmp = 10 ** (amplitude / 20)
        freq = np.linspace(this.freq, next.freq, 1024)
        sinoide = linAmp * np.sin(2 * np.pi * freq * np.arange(1024) / sr + phrases)
        signal += sinoide
    pd.set_obj_var("index", index + 1)
    
    return signal

def py4pd_spt_setup():
    pd.add_object(stft_tab, "spt.stft", py_out=True)

    # Peak Tracking
    pd.add_object(spear_pt, "spt.pt", py_out=True)
    pd.add_object(spear_pt_live, "spt.pt-live", py_out=True)

    # Synth
    pd.add_object(synth, "spt.synth", py_out=True)
    pd.add_object(speat_livesynth, "spt.synth~", py_out=True, obj_type=pd.AUDIOOUT)


