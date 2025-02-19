import os
import numpy as np
from sklearn.metrics import accuracy_score
from Model import *
from torch.utils.data import DataLoader, random_split
from data_load import ImageDataset
from resnet50_spd import *

# Weight Decay 적용
# retain_graph 제거
# num_workers 설정
# schedular 추가
# train에 zero grad가 두번 잇어서 하나 지웢줌줌
# ce, mse loss 가 두번 계산되어 지워줌줌


# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 4096, 4096
CNN_embed_dim = 128  # latent dim extracted by 2D CNN
dropout_p = 0.3  # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 2
RNN_hidden_nodes = 128
RNN_FC_dim = 256

alpha = 1
beta = 0.0005

# training parameters
epochs = 10000  # training epochs
batch_size = 256
learning_rate = 1e-4
log_interval = 8  # interval for displaying training info

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(use_cuda)
save_model_path = './ckpt'

cnn_encoder = resnet50().to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p).to(device)
pred_class_model = PredictLabelModel().to(device)
pred_sheer_model = PredictSheerModel().to(device)

train_dataset = ImageDataset("./data/train")
validation_dataset = ImageDataset("./data/validation")
test_dataset = ImageDataset("./data/test")

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)


def train(log_interval, model, device, train_loader, crnn_optimizer, classification_optimizer, regression_optimizer, scheduler, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder, classification_model, regression_model = model
    cnn_encoder.train()
    rnn_decoder.train()
    pred_class_model.train()
    pred_sheer_model.train()

    losses1 = []
    losses2 = []
    scores = []
    N_count = 0  # counting total trained sample in one epoch

    

    for batch_idx, (X, y, sheer) in enumerate(train_loader):
    # distribute data to device
        X, y, sheer = X.to(device), y.to(device), sheer.to(device)
        

        N_count += X.size(0)

        crnn_optimizer.zero_grad()
        classification_optimizer.zero_grad()
        regression_optimizer.zero_grad()

        rnn_output = rnn_decoder(cnn_encoder(X))
        classification_output =  pred_class_model(rnn_output) # output has dim = (batch, number of classes)
        regression_output = pred_sheer_model(rnn_output)

        CEloss = torch.nn.functional.cross_entropy(classification_output.float(), y.float())
        losses1.append(CEloss.item())

        # to compute accuracy
        y = torch.max(y, 1)[1]
        y_pred = torch.max(classification_output, 1)[1]  # y_pred != output

        MSEloss = torch.nn.functional.mse_loss(regression_output.float(), sheer.float(), reduction='mean')

        losses2.append(MSEloss.item())

        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)  # computed on CPU

        MSEloss *= beta
        loss = (alpha * CEloss + MSEloss) / (alpha + beta)
        loss.backward()
        crnn_optimizer.step()

        #CEloss.backward()
        classification_optimizer.step()

        #MSEloss.backward()
        regression_optimizer.step()

        

        # show information
        if (batch_idx + 1) % log_interval == 0:
            if (batch_idx + 1) % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%\tLoss of sheer: {:.6f}'.format(
                    epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                    CEloss.item(),
                    100 * step_score, MSEloss.item()))
                

        

    return losses1, losses2, scores


def validation(model, device, crnn_optimizer, classification_optimizer, regression_optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder, classification_model, regression_model = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    classification_model.eval()
    regression_model.eval()

    test_CE_loss = 0
    test_MSE_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y, sheer in test_loader:
            # distribute data to device
            X, y, sheer = X.to(device), y.to(device), sheer.to(device)

            rnn_output = rnn_decoder(cnn_encoder(X))
            classification_output = pred_class_model(rnn_output)  # output has dim = (batch, number of classes)
            regression_output = pred_sheer_model(rnn_output)

            loss = torch.nn.functional.cross_entropy(classification_output.float(), y.float(), reduction='sum')
            test_CE_loss += loss.item()  # sum up batch loss

            loss2 = torch.nn.functional.mse_loss(regression_output.float(), sheer.float(), reduction='sum')
            test_MSE_loss += loss2.item()

            y = torch.max(y, 1)[1]
            y_pred = classification_output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_CE_loss /= len(test_dataset)
    test_MSE_loss /= (len(test_dataset) * 2)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average CE loss: {:.4f}, Accuracy: {:.2f}%'.format(
        len(all_y), test_CE_loss * alpha,
        100 * test_score))
    print('Test set ({:d} samples): Average MSE loss: {:.4f}\n'.format(
        len(all_y), test_MSE_loss * beta))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(),
               os.path.join(save_model_path + 'cnn_encoder_epoch.pth'))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(),
               os.path.join(save_model_path + 'rnn_decoder_epoch.pth'))  # save motion_encoder
    torch.save(classification_model.state_dict(),
               os.path.join(save_model_path + 'pred_class_epoch.pth'))
    torch.save(regression_model.state_dict(),
               os.path.join(save_model_path + 'pred_sheer_epoch.pth'))
    torch.save(crnn_optimizer.state_dict(),
               os.path.join(save_model_path + 'crnn_optimizer_epoch{}.pth'))  # save optimizer
    torch.save(classification_optimizer.state_dict(),
               os.path.join(save_model_path + 'classification_optimizer_epoch{}.pth'))
    torch.save(regression_optimizer.state_dict(),
               os.path.join(save_model_path + 'regression_epoch{}.pth'))
    print("Epoch {} model saved!".format(epoch + 1))

    return test_CE_loss, test_MSE_loss, test_score


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)
    pred_class_model = nn.DataParallel(pred_class_model)
    pred_sheer_model = nn.DataParallel(pred_sheer_model)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.parameters()) + \
                  list(rnn_decoder.parameters())

    classification_params = list(pred_class_model.parameters())
    regression_params = list(pred_sheer_model.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.parameters()) + \
                  list(rnn_decoder.parameters())

    classification_params = list(pred_class_model.parameters())
    regression_params = list(pred_sheer_model.parameters())

#옵티마이저 수정!!!!!!!!!!!!!!!!

weight_decay = 1e-4  # Weight Decay 설정

# BN 제외하고 Weight Decay 적용
crnn_params = [
    {"params": [p for n, p in cnn_encoder.named_parameters() if "bn" not in n], "weight_decay": weight_decay},
    {"params": [p for n, p in cnn_encoder.named_parameters() if "bn" in n], "weight_decay": 0.0},
    {"params": [p for n, p in rnn_decoder.named_parameters() if "bn" not in n], "weight_decay": weight_decay},
    {"params": [p for n, p in rnn_decoder.named_parameters() if "bn" in n], "weight_decay": 0.0},
]

classification_params = [
    {"params": [p for n, p in pred_class_model.named_parameters() if "bn" not in n], "weight_decay": weight_decay},
    {"params": [p for n, p in pred_class_model.named_parameters() if "bn" in n], "weight_decay": 0.0},
]

regression_params = [
    {"params": [p for n, p in pred_sheer_model.named_parameters() if "bn" not in n], "weight_decay": weight_decay},
    {"params": [p for n, p in pred_sheer_model.named_parameters() if "bn" in n], "weight_decay": 0.0},
]

# 옵티마이저 변경 (RMSprop -> AdamW)
crnn_optimizer = torch.optim.AdamW(crnn_params, lr=learning_rate)
classification_optimizer = torch.optim.AdamW(classification_params, lr=learning_rate)
regression_optimizer = torch.optim.AdamW(regression_params, lr=learning_rate)

# 학습률 스케줄러 정의 (모든 옵티마이저에 적용)
crnn_scheduler = torch.optim.lr_scheduler.StepLR(crnn_optimizer, step_size=10, gamma=0.1)
classification_scheduler = torch.optim.lr_scheduler.StepLR(classification_optimizer, step_size=10, gamma=0.1)
regression_scheduler = torch.optim.lr_scheduler.StepLR(regression_optimizer, step_size=10, gamma=0.1)


# record training process
epoch_train_CE_losses = []
epoch_train_MSE_losses = []
epoch_train_scores = []

epoch_test_CE_losses = []
epoch_test_MSE_losses = []
epoch_test_scores = []

# 학습률 스케줄러 정의 (학습률 감소 주기: 10 epoch마다, 감소율: 0.1)
scheduler = [crnn_scheduler,classification_scheduler,regression_scheduler]

# start training
for epoch in range(epochs):
    # train, test model
    train_CE_losses, train_MSE_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder, pred_class_model, pred_sheer_model], device, train_dataloader,
                                                            crnn_optimizer, classification_optimizer, regression_optimizer,
                                       scheduler,epoch)
    epoch_test_CE_loss, epoch_test_MSE_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder, pred_class_model, pred_sheer_model], device,
                                                                           crnn_optimizer, classification_optimizer, regression_optimizer, validation_dataloader)

    # ✅ 모든 옵티마이저의 스케줄러 업데이트
    crnn_scheduler.step()
    classification_scheduler.step()
    regression_scheduler.step()

    # save results
    epoch_train_CE_losses.append(train_CE_losses)
    epoch_train_MSE_losses.append(train_MSE_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_CE_losses.append(epoch_test_CE_loss)
    epoch_test_MSE_losses.append(epoch_test_MSE_loss)
    epoch_test_scores.append(epoch_test_score)



    # save all train test results
    A = np.array(epoch_train_CE_losses)
    B = np.array(epoch_train_MSE_losses)
    C = np.array(epoch_train_scores)
    D = np.array(epoch_test_CE_losses)
    E = np.array(epoch_test_MSE_losses)
    F = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_CE_losses.npy', A)
    np.save('./CRNN_epoch_training_MSE_losses.npy', B)
    np.save('./CRNN_epoch_training_scores.npy', C)
    np.save('./CRNN_epoch_test_CE_loss.npy', D)
    np.save('./CRNN_epoch_test_MSE_loss.npy', E)
    np.save('./CRNN_epoch_test_score.npy', F)