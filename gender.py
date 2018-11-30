import parselmouth
import glob
import math
import os.path

# filename -> pitch values
male_data = {}
female_data = {}

def extract_pitch(filepath, db):
    for filepath in glob.iglob("male/*.mp3"):
        for file in glob.glob(filepath):
            s = parselmouth.Sound(file)

        pitch = s.to_pitch()
        duration = pitch.get_total_duration()

        values = []
        frames = 4000
        for i in range(1, frames):
            frame = i/float(frames)
            value = pitch.get_value_at_time(frame * duration)
            if not math.isnan(value):
                values.append(value)

        print len(values)
        db[filepath] = values

extract_pitch("female/*.mp3", female_data)
extract_pitch("male/*.mp3", male_data)

print male_data
print female_data
