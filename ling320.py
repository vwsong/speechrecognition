import parselmouth
import glob
import math
import numpy as np
import os.path

FORMANT_1 = 0
FORMANT_2 = 1
PITCH = 2

# filename -> pitch values
male_data = {}
female_data = {}

def extract_features(fp, db):
    for filepath in glob.iglob(fp):
        for file in glob.glob(filepath):
            s = parselmouth.Sound(file)

        pitch = s.to_pitch()
        formant = s.to_formant_burg()
        duration = pitch.get_total_duration()

        values = []
        frames = 400
        for i in range(1, frames):
            frame = i/float(frames)
            pitch = pitch.get_value_at_time(frame * duration)
            f1 = formant.get_value_at_time(1)
            f2 = formant.get_value_at_time(2)

            if not math.isnan(pitch) and not math.isnan(f1) and not math.isnan(f2):
                features = [f1, f2, pitch]
                values.append(features)

        db[filepath] = values

extract_features("female/*.mp3", female_data)
extract_features("male/*.mp3", male_data)
