import torch
import os

audio_samples = 'dataset/audio_samples'
video_frames = 'dataset/video_clips_096'


# for i in os.listdir(audio_samples):
#     for sample in os.listdir(os.path.join(audio_samples, i)):
#         audio_sample = torch.load(os.path.join(audio_samples, i, sample))
#         sample_name = sample.replace('.pt', '')
#         video_sample = len(os.listdir(os.path.join(video_frames, i, sample_name)))
#         if (len(audio_sample)+1) != (video_sample):
#             print(len(audio_sample))
#             print('audio num: ', os.path.join(audio_samples, i, sample))
#             print(video_sample)
#             print('video num: ', os.path.join(video_frames, i, sample_name))


# k = 0
# for i in os.listdir(video_frames):
#     for sample in os.listdir(os.path.join(video_frames, i)):
#         k += 1

# print(k)


result_csv = 'results/audio/cmd3_audio_last_0305/result_reports.csv'

clip_0 = 0 
clip_1 = 0
clip_2 = 0

with open(result_csv) as f:
    lines = f.read().splitlines()
for line in lines:
    _, _, filename, label, pred = line.split(',')
    if label == "0":
        clip_0 += 1
    elif label == "1":
        clip_1 += 1
    elif label == "2":
        clip_2 += 1

print('0: ', clip_0)
print('1: ', clip_1)
print('2: ', clip_2)

print("0 (%)", clip_0 / (clip_0 + clip_1 + clip_2))
print("1 (%)", clip_1 / (clip_0 + clip_1 + clip_2))
print("2 (%)", clip_2 / (clip_0 + clip_1 + clip_2))

