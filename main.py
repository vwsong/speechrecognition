import parselmouth
import glob
import os.path
import numpy as np
#from sklearn.mixture import GMM

## TODO:
# load in all audio files
# figure out how to cut off last sentence for testing (or a small phrase)
# figure out how to plot mfcc using api
# visualize gaussians
    # is it fast enough
# once all done see if it works live


values = []

for fp in glob.iglob("male/*.mp3"):
    for file in glob.glob(fp):
        s = parselmouth.Sound(file)

    mfcc = s.to_mfcc()
    features = mfcc.extract_features()
    values.append(features.as_array())

for fp in glob.iglob("female/*.mp3"):
    for file in glob.glob(fp):
        s = parselmouth.Sound(file)

    mfcc = s.to_mfcc()
    features = mfcc.extract_features()
    values.append(features.as_array())
