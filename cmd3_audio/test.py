from models.vggish import VGGish
import torch.nn.functional as F
import torch.nn as nn


model_urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

model = VGGish(urls=model_urls)
print(model)
model.eval()

print(model.embeddings[4])
model.embeddings[4] = nn.Linear(4096, 3)
model.cuda()
print(model)

outputs = model("./bus_chatter.wav")
print(outputs)
print(outputs.shape)

# outputs = F.softmax(outputs, dim=1)
# print(outputs)
# print(outputs.shape)