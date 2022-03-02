import os

path = 'dataset/video_clips_096'

x = 0
for i in os.listdir(path):
    for j in os.listdir(os.path.join(path, i)):
        x += 1
        # for k in os.listdir(os.path.join(path, i, j)):
            # x += 1
        
print(x)