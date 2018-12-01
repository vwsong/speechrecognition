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


for file in glob.glob("english1.mp3"):
    s = parselmouth.Sound(file)

mfcc = s.to_mfcc()
features = mfcc.extract_features()
print features.to_matrix_features()
#print features.as_array()
