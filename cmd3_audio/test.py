# from models.vggish import VGGish
# import torch.nn.functional as F
# import torch.nn as nn

# from data.dataset import CMD3Dataset
# import torchaudio
# import torchaudio.transforms as T

# # data_path = "cmd3_audio/dataset/set_splits/train_split.csv"

# # dataset = CMD3Dataset(data_path, audio_path)
# # print(len(dataset))


# # model_path = "cmd3_audio/models/vggish_pretrained.pth"

# # model = VGGish(model_path, 3)
# # print(model)
# # model.eval()


# audio_path = "cmd3_audio/dataset/audio/2017/-Koj9hvcBMk.wav"

# waveform, sample_rate = torchaudio.load(audio_path)

# print(waveform.shape)
# print(sample_rate)

# transform = T.MelSpectrogram(sample_rate=sample_rate,
#                              n_fft=960,
#                              n_mels=64,
#                              win_length=25,
#                              hop_length=10,)
# mel_specgram = transform(waveform)
# print(mel_specgram.shape)



import wandb

wandb.init(name = "cmd3_logger", project="director_classification")