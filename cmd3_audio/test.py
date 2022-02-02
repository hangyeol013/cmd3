from models.vggish import VGGish
import torch.nn.functional as F
import torch.nn as nn

from data.dataset import CMD3Dataset

data_path = "cmd3_audio/dataset/set_splits/train_split.csv"
audio_path = "cmd3_audio/dataset/audio"

dataset = CMD3Dataset(data_path, audio_path)
print(len(dataset))


model_urls = "cmd3_audio/models/vggish_pretrained.pth"

model = VGGish(model_urls)
# print(model)
# model.eval()

# print(model.embeddings[4])
# model.embeddings[4] = nn.Linear(4096, 3)
# model.cuda()
# print(model)

# outputs = model("./bus_chatter.wav")
# print(outputs)
# print(outputs.shape)

# outputs = F.softmax(outputs, dim=1)
# print(outputs)
# print(outputs.shape)