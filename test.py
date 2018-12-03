import python_speech_features as mfcc
import scipy.io.wavfile as wavfile
import glob
import os.path

# (rate,sig) = wav.read("male/english1.mp3")

# d_mfcc_feat = delta(mfcc_feat, 2)
# fbank_feat = logfbank(sig,rate)

# print(fbank_feat[1:3,:])
mfccs = []
for wav in glob.iglob("wav/*/wav/*.wav"):
    (rate,audio) = wavfile.read(wav)
    mfcc_feat = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, appendEnergy = True)
    mfccs.append(mfcc_feat)
