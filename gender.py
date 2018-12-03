import parselmouth
import glob
import math
import numpy as np
import os.path

# filename -> pitch values
male_data = {}
female_data = {}

def extract_pitch(fp, db):
    for filepath in glob.iglob(fp):
        for file in glob.glob(filepath):
            s = parselmouth.Sound(file)

        pitch = s.to_pitch()
        duration = pitch.get_total_duration()

        values = []
        frames = 400
        for i in range(1, frames):
            frame = i/float(frames)
            value = pitch.get_value_at_time(frame * duration)
            if not math.isnan(value):
                values.append(value)

        db[filepath] = values

extract_pitch("female/*.mp3", female_data)
extract_pitch("male/*.mp3", male_data)

male_values = []
female_values = []

for key in male_data.keys():
    value = male_data[key]
    male_values += value

for key in female_data.keys():
    value = female_data[key]
    female_values += value

print np.average(male_values)
print np.std(male_values)

print np.average(female_values)
print np.std(female_values)
