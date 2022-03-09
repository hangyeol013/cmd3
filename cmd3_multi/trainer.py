import torch
import torch.nn as nn
import numpy as np

from torch.utils import data


NUM_CLASSES = 3

def feature_fuse_train(vfeature_path, afeature_path, targets_path, ids_path):
    
    v_features = np.load(vfeature_path, allow_pickle=True)
    a_features = np.load(afeature_path, allow_pickle=True)
    concat_features = torch.cat((v_features, a_features), 1)

    ids = np.load(ids_path, allow_pickle=True)
    targets = np.load(targets_path, allow_pickle=True)

    model = nn.Linear(concat_features.shape[1], NUM_CLASSES).cuda()
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    print('feature size: ', concat_features.shape)
    print('target size: ', targets.shape)

    train_loader = data.DataLoader(data.TensorDataset(concat_features, targets), batch_size=16, shuffle=True)

    n_epochs = 1
    for epoch in range(n_epochs):
        for batch in train_loader:
            
            inputs = batch[0].float()
            inputs = inputs.cuda()
            targets = batch[1].cuda()
 
            preds = model(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            print('loss: ', loss)

    torch.save(model.state_dict(), 'cmd3_multi/cmd3_multimodal.pth')


if __name__ == '__main__':
    
    fuse_mode = 'feature'
    
    # Train
    vfeatures_train_path = 'custom_features/video/cmd3_video_last_trainfeature_0308/features.pk'
    afeatures_train_path = 'custom_features/audio/cmd3_audio_last_trainfeature_0308/features.pk'
    targets_train_path = 'custom_features/audio/cmd3_audio_last_trainfeature_0308/targets.pk'
    ids_train_path = 'custom_features/audio/cmd3_audio_last_trainfeature_0308/ids.pk'


    feature_fuse_train(vfeatures_train_path, afeatures_train_path, targets_train_path, ids_train_path)