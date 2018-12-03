import parselmouth
import glob
import math
import random
import numpy as np
import os.path
from functools import reduce
from scipy.spatial import distance

FORMANT_1 = 0
FORMANT_2 = 1
PITCH = 2
TRAIN_COEFFICIENT = 0.75
VERIFY_COEFFICIENT = 0.25

# filename -> (pitch values, mean)
data = {}
datapoints = {}

def get_mean(values):
    count = len(values)
    f1 = 0
    f2 = 0
    p = 0

    for i in values:
        f1 += i[FORMANT_1]
        f2 += i[FORMANT_2]
        p += i[PITCH]

    mean = (f1/count, f2/count, p/count)
    return mean

def train(fp, db):
    for filepath in glob.iglob(fp):
        for file in glob.glob(filepath):
            s = parselmouth.Sound(file)

        pitch = s.to_pitch()
        formant = s.to_formant_burg()
        duration = pitch.get_total_duration()

        values = []
        frames = 400
        if file not in datapoints:
            datapoints[file] = set()
        for i in range(1, int(frames * TRAIN_COEFFICIENT)):
            r = random.randint(1, frames)
            datapoints[file].add(r)
            frame = r/float(frames)
            time = frame * duration
            p = pitch.get_value_at_time(time)
            f1 = formant.get_value_at_time(1, time)
            f2 = formant.get_value_at_time(2, time)

            if not math.isnan(p) and not math.isnan(f1) and not math.isnan(f2):
                features = (f1, f2, p)
                values.append(features)

        # print len(values)
        mean = get_mean(values)
        db[filepath] = (values, mean)

def predict(mean, data):
    hiscore = float('inf')
    output = None
    for key in data.keys():
        value = data[key][1]
        dst = distance.euclidean(value, mean)
        if dst < hiscore:
            output = key
            hiscore = dst

    return output

def test(fp):
    correct = 0
    total = 0
    for filepath in glob.iglob(fp):
        for file in glob.glob(filepath):
            s = parselmouth.Sound(file)

        pitch = s.to_pitch()
        formant = s.to_formant_burg()
        duration = pitch.get_total_duration()

        values = []
        frames = 400
        for i in range(1, int(frames * VERIFY_COEFFICIENT)):
            r = random.randint(1, frames)
            frame = r/float(frames)
            time = frame * duration
            p = pitch.get_value_at_time(time)
            f1 = formant.get_value_at_time(1, time)
            f2 = formant.get_value_at_time(2, time)

            if not math.isnan(p) and not math.isnan(f1) and not math.isnan(f2):
                features = (f1, f2, p)
                values.append(features)

        mean = get_mean(values)
        total += 1
        correct = correct + 1 if predict(mean, data) == filepath else correct

    return (correct, total)

train("female/*.mp3", data)
train("male/*.mp3", data)
print test("female/*.mp3")
print test("male/*.mp3")
