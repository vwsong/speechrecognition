import parselmouth
import glob
import os.path

values = []

# for fp in glob.iglob("male/*.mp3"):
#     for file in glob.glob(fp):
#         s = parselmouth.Sound(file)
#
#     mfcc = s.to_mfcc()
#     features = mfcc.extract_features()
    # values.append(features.as_array())

for fp in glob.iglob("female/*.mp3"):
    for file in glob.glob(fp):
        s = parselmouth.Sound(file)

    mfcc = s.to_mfcc()
    features = mfcc.extract_features(window_length=1.0)
    values.append(features.as_array())

print values[0]
