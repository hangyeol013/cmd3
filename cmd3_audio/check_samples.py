import torch
import os

audio_samples = 'dataset/audio_samples'
video_frames = 'dataset/video_clips_096'


for i in os.listdir(audio_samples):
    for sample in os.listdir(os.path.join(audio_samples, i)):
        audio_sample = torch.load(os.path.join(audio_samples, i, sample))
        sample_name = sample.replace('.pt', '')
        video_sample = len(os.listdir(os.path.join(video_frames, i, sample_name)))
        if (len(audio_sample)+1) != (video_sample):
            print(len(audio_sample))
            print('audio num: ', os.path.join(audio_samples, i, sample))
            print(video_sample)
            print('video num: ', os.path.join(video_frames, i, sample_name))


k = 0
for i in os.listdir(video_frames):
    for sample in os.listdir(os.path.join(video_frames, i)):
        k += 1

print(k)