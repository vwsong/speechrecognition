import parselmouth
import glob
import os.path

for file in glob.glob("english1.mp3"):
    s = parselmouth.Sound(file)

mfcc = s.to_mfcc()
features = mfcc.extract_features()
print features.as_array()
