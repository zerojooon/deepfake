import pickle

import cv2
import numpy as np
import models
import simple_model
import face_detector_1
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import time
import logging
from mtcnn import mtcnn

# 바꾼 부분 1 : Multi GPU + VGG -> cuda:0  global_model -> cuda:1
# 바꾼 부분 2 : ResNet, global model 의 마지막 layer 에 sigmoid 추가
# 바꾼 부분 3 : answer 이 REAL 일 때 [1.0], FAKE 일 때 [0.0] 값 넣어줌.
# 바꾼 부분 4 : optimizer Adam -> SGD
# 바꾼 부분 5 : global model => CONV1D, CONV2D
# 바꾼 부분 6 : loss MSE -> BCE
# 바꾼 부분 7 : print -> log 파일로 저장


detector = mtcnn.MTCNN()  # Face detection model

#Argument Paster
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--lr_global', type=float, default=0.001, help='Learning Rate Decrease Ratio per a Epoch')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
parser.add_argument('--network', type=str, default='resnet18', help='Network Setlection')
parser.add_argument('--global_network', type=str, default='Conv2D', help='global network selection')
parser.add_argument('--crop', type=bool, default=True, help='feed cropped image from FaceNet')
parser.add_argument('--save', type=bool, default=True, help='Whether save trained models')
parser.add_argument('--force_face_load', type=bool, default=False, help='force face load')
opt = parser.parse_args()


# hyperparameters
num_frames = 100
kernel_size = 5 # Conv2D global model
h_size = 224
v_size = 224
slide_size = 50

# 학습 Setup
device = torch.device("cuda" if opt.cuda else "cpu")
model = models.resnet18()
criterion = nn.BCELoss()


# 여기서 전체 frame을 concat한 것 을 입력으로 최종 Video의 decision 하는 model을 만들 필요 있음
global_models = {'Conv1D' : simple_model.Conv1D_Net, 'Conv2D' : simple_model.Conv2D_Net}
model_global = global_models[opt.global_network](inplane=slide_size)


# memory allocate
if torch.cuda.device_count() > 1:
   print("use multiple GPUs")
   model = nn.DataParallel(model, device_ids=[0]).cuda()
   model_global = nn.DataParallel(model_global, device_ids=[1]).cuda()
model = model.to('cuda:0')
model_global = model_global.to('cuda:1')


# DATA Loader
save_dir = './result/'    #save cropped image numpy array in pickle file
train_dir = './data/train_sample_videos/'
train_video_files = [train_dir + x for x in os.listdir(train_dir) if x.endswith('.mp4')]
valid_dir = './data/valid/'
valid_video_files = [valid_dir + x for x in os.listdir(valid_dir)]
train_metadata = pd.read_json(train_dir+'metadata.json')
train_metadata = train_metadata.T
train_metadata.head()
train_metadata['label'].value_counts(normalize=True)
print(train_metadata)


#======get real/false file_name=========
real_list = []
fake_list = []
for video in train_video_files:
   answer = train_metadata.loc[os.path.basename(video), 'label']
   if answer == 'REAL':
       real_list.append(os.path.basename(video))
   else:
       fake_list.append(os.path.basename(video))




# # logging
# logging.basicConfig(
#     level=logging.INFO,
#     filename='./'+time.strftime("%Y%m%d-%H%M%S")+'.log',
# )
# logger = logging.getLogger('./train.log')
# logger.info(opt)

# Traning
model.train()
model_global.train()

init_frames = 10
save_flag = opt.force_face_load


for epoch in range(opt.nEpochs):
    epoch_loss = 0.0

    f_lr_decayed = opt.lr * (0.9 ** epoch)
    g_lr_decayed = opt.lr_global * (0.9 ** epoch)

    optimizer_frame = optim.SGD(model.parameters(), lr=f_lr_decayed)
    optimizer_global = optim.SGD(model_global.parameters(), lr=g_lr_decayed)

    print('============ Epoch [{:3d}] ============ : lr={:0.7f} glr={:0.7f}'.format(epoch, f_lr_decayed, g_lr_decayed))

    #real video detection
    i = 0  #index of real video list
    j = 0  #index of fake video_list



    while(j < len(fake_list)):
        GLOBAL_TRAIN = False  # flag putting slides into global model
        out_tensor = torch.tensor([[], []]).to(device)
        fc_list = []  # list of fully connected layers --used for sliding task

        batch = np.zeros((2, 3, h_size, v_size))
        batch = torch.from_numpy(batch).float().to(device)

        if i == len(real_list):   # num of real video is smaller than num of fake video, so go back.
           i = 0
        real_file = real_list[i]
        fake_file = fake_list[j]

        # 이 이후 부분을 파일마다 읽어서 처리하는 형태로 변경 필요
        # movie_path = './video/train/aagfhgtpmv.mp4'
        real_capture = cv2.VideoCapture(train_dir+real_file)
        fake_capture = cv2.VideoCapture(train_dir+fake_file)
        real_width = real_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        real_height = real_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fake_width = fake_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        fake_height = fake_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        real_ans = torch.tensor([1.0]).to(device)  #answer for real video
        fake_ans = torch.tensor([0.0]).to(device)  #answer  for fake video
        ans_batch = torch.cat((fake_ans, real_ans), 0)
        # ans_batch = torch.unsqueeze(ans_batch, 1)
        #print(ans_batch.shape)

        # 전체 frame 의 FC를 concat 하는 텐서
        #out_tensor = torch.tensor([]).to(device)

        # face_detector 에 face_box 를 넣어 그 전 프레임의 위치와 비교
        real_mid_x = (real_width -v_size) / 2
        real_mid_y = (real_height - h_size) / 2
        fake_mid_x = (fake_width - v_size) / 2
        fake_mid_y = (fake_height - h_size) / 2

        real_face_box = (real_mid_x, real_mid_y, h_size, v_size)
        fake_face_box = (fake_mid_x, fake_mid_y, h_size, v_size)

        # avg_face_accu = 0
        # n_detect_failed = 0
        # n_center_loc = 0
        # n_face_moved = 0
        # avg_detect = 0
        # iteration = 0

        save_real_list = []
        save_fake_list = []

        decision_fake_sum = 0
        decision_real_sum = 0
        decision_n = 0

        for iteration in range(0, num_frames + init_frames):
            load_real_flag = False
            load_fake_flag = False
            real_flag = True
            fake_flag = True
            _, real_frame = real_capture.read()
            _, fake_frame = fake_capture.read()


            if opt.crop:
                if iteration < init_frames:
                    run_mode = 0     # frame<10
                else:
                    run_mode = 1    # frame >=10

                if real_file.split(".")[0]+".pkl" in os.listdir(save_dir):
                    load_real_flag = True
                    # print(str(iteration), real_file, "real_file in save_dir")

                    real_file_name = os.path.join(save_dir, real_file.split('.')[0]) + '.pkl'
                    with open(real_file_name, 'rb') as f:
                        real_face = pickle.load(f)[iteration]


                if fake_file.split(".")[0]+".pkl" in os.listdir(save_dir):
                    load_fake_flag = True
                    # print(str(iteration), fake_file, "fake_file in save_dir")

                    fake_file_name = os.path.join(save_dir, fake_file.split('.')[0]) + '.pkl'
                    with open(fake_file_name, 'rb') as f:
                        fake_face = pickle.load(f)[iteration]



                if not load_real_flag:
                    real_face_loc, real_face, real_detect_acc, real_detect_failed, real_center_loc, real_face_moved = face_detector_1.crop_face(
                        detector, real_frame, real_face_box, h_size, v_size, real_width, real_height, ALPHA=0.5, run_mode=run_mode)

                    if real_face.any() == None:
                        save_real_list.append(real_frame)
                        print("real image is NOne")

                    else:
                        real_face_box = real_face_loc
                        real_frame = real_face
                        save_real_list.append(real_face)

                    if iteration == (num_frames + init_frames - 1):
                        real_file_name = os.path.join(save_dir, real_file.split('.')[0]) + '.pkl'
                        print("saving real list into directory: " + str(len(save_real_list)))
                        with open(real_file_name, 'wb') as f:
                            pickle.dump(save_real_list, f)


                if not load_fake_flag:
                    #print(fake_file)
                    fake_face_loc, fake_face, fake_detect_acc, fake_detect_failed, fake_center_loc, fake_face_moved = face_detector_1.crop_face(
                        detector, fake_frame, fake_face_box, h_size, v_size, fake_width, fake_height,ALPHA=0.5, run_mode=run_mode)

                    fake_file_name = os.path.join(save_dir, fake_file.split('.')[0])+'.pkl'


                    if fake_face.any() == None:
                        save_fake_list.append(fake_frame)
                        fake_flag = False
                        print("fake image is none")

                    else:
                        fake_face_box = fake_face_loc
                        fake_frame = fake_face
                        save_fake_list.append(real_face)

                    if iteration == (num_frames + init_frames - 1):
                        fake_file_name = os.path.join(save_dir, fake_file.split('.')[0]) + '.pkl'
                        print("saving fake list into directory: " + str(len(save_fake_list)))
                        with open(fake_file_name, 'wb') as f:
                            pickle.dump(save_fake_list, f)



                # if iteration % 10 == 1:
                #     print(
                #            "===> Epoch[{:02d}], Video_name:({},{}) Video_idx:{:3d}/{:3d}, ({:4d}/{:4d})".format(
                #                epoch, real_file, fake_file, j + 1,
                #                len(train_video_files),
                #                iteration, num_frames + 1))


            if run_mode == 1:
               real_frame = cv2.resize(real_frame, (h_size, v_size), interpolation=cv2.INTER_LINEAR)
               fake_frame = cv2.resize(fake_frame, (h_size, v_size), interpolation=cv2.INTER_LINEAR)

               real_frame = torch.autograd.Variable(torch.from_numpy(real_frame).permute(2, 0, 1))
               fake_frame = torch.autograd.Variable(torch.from_numpy(fake_frame).permute(2, 0, 1))

               # real_frame=real_frame.permute(2,0,1)
               # print('frame',real_frame.shape)
               # print('batch:',batch.shape)
               # print(iteration)

               # batch = np.swapaxes(batch, 1, 3)
               # batch = torch.from_numpy(batch).float().to(device)

               batch[0, :, :, :] = fake_frame
               batch[1, :, :, :] = real_frame

               optimizer_frame.zero_grad()
               decision, fc_out = model(batch)
               # print("decision : ", decision.shape)
               # print("fc : ", fc_out.shape)
               # out_tensor = torch.cat((out_tensor, fc_out), 1)

               loss = criterion(decision, ans_batch.cuda(0))
               epoch_loss += loss.item()
               loss.backward()

               optimizer_frame.step()

               # avg_detect += decision.item()
               if (iteration-init_frames < slide_size):
                   fc_list.append(fc_out.unsqueeze(dim=1))
                   # print(iteration)
                   # print(len(fc_list))
                   if iteration-init_frames == slide_size - 1:
                       GLOBAL_TRAIN = True
               else:
                   fc_list = fc_list[1:]
                   fc_list.append(fc_out.unsqueeze(dim=1))

               # if not GLOBAL_TRAIN and iteration % 10 == 1:
               #     print(
               #         "===> Epoch[{:02d}] Video_idx:{}/{}, ({:4d}/{:4d}): Loss: {:.4f}".format(epoch, j + 1,
               #                                                                                  len(
               #                                                                                      train_video_files),
               #                                                                                  iteration-init_frames,
               #                                                                                  num_frames,
               #                                                                                  loss.item()))
               #print(GLOBAL_TRAIN)

               decision_n += 1
               decision_fake_sum += decision[0].item()
               decision_real_sum += decision[1].item()

               if GLOBAL_TRAIN:
                   # print("global model working  ",iteration)
                   slide_tensor = torch.cat(fc_list, dim=1)

                   optimizer_global.zero_grad()

                   # # runtime error (buffers been freed) 로 인하여 detach 를 해서 loss 흘릴 때 게산하지 않음
                   # print(slide_tensor.shape)
                   global_fc_input = slide_tensor #.permute(1, 0, 2)
                   # print(global_fc_input.shape)
                   global_fc_input = global_fc_input.detach().cuda(1)
                   global_fc_input = global_fc_input[:, None, :, :]
                   final_decision = model_global(global_fc_input)

                   # loss_global = criterion(final_decision.squeeze(0), ans_batch.cuda(1))
                   loss_global = criterion(final_decision, ans_batch.cuda(1))
                   loss_global.backward()
                   optimizer_global.step()

                   if iteration % 10 == 1:
                       print(
                           "===> Epoch[{:02d}] Video_name:({},{}), Video_idx:{:3d}/{:3d}, ({:4d}/{:4d}): Frame[Loss: {:.4f}, Fake_avg:{:.4f} Real_avg:{:.4f}] Global[Loss: {:.4f}, Fake_avg:{:.4f}, Real_avg:{:.4f}]".format(
                               epoch, real_file, fake_file, j + 1,
                               len(fake_list),
                               iteration, num_frames + 1,
                               loss.item(), decision_fake_sum / decision_n, decision_real_sum / decision_n,
                               loss_global.item(), final_decision[0].item(), final_decision[1].item())
                       )
                       decision_n = 0
                       decision_fake_sum = 0.0
                       decision_real_sum = 0.0

               else:
                   if iteration % 10 == 1:
                       print(
                           "===> Epoch[{:02d}] Video_name:({},{}), Video_idx:{:3d}/{:3d}, ({:4d}/{:4d}): Frame[Loss: {:.4f}, Fake_avg:{:.4f} Real_avg:{:.4f}] ".format(
                               epoch, real_file, fake_file, j + 1,
                               len(fake_list),
                               iteration+1, num_frames,
                               loss.item(), decision_fake_sum / decision_n, decision_real_sum / decision_n))
                       decision_n = 0
                       decision_fake_sum = 0.0
                       decision_real_sum = 0.0

        i += 1
        j += 1

    # SAVE MODEL
    # if opt.save:
    #     model_out_path = "pretrain/0224_frame_model_epoch_{}.pth".format(epoch)
    #     torch.save(model, model_out_path)
    #     print("Checkpoint saved to {}".format(model_out_path))
    #
    #     model_out_path2 = "pretrain/0224_global_model_epoch_{}.pth".format(epoch)
    #     torch.save(model_global, model_out_path2)
    #     print("Checkpoint saved to {}".format(model_out_path2))

'''
       # face_detector 에 face_box 를 넣어 그 전 프레임의 위치와 비교
       face_box = (846, 426, h_size, v_size)


       optimizer_global.zero_grad()

       # runtime error (buffers been freed) 로 인하여 detach 를 해서 loss 흘릴 때 게산하지 않음
       global_fc_input = out_tensor[None, :, :]
       global_fc_input = global_fc_input.detach().cuda(1)
       final_decision = model_global(global_fc_input)

       loss_global = criterion(final_decision.squeeze(0), ans_batch.cuda(1))
       loss_global.backward()
       optimizer_global.step()

       video_info = "{:s}({})({:3d}/{:3d})=> DeepFake[Frame_avg:{:1.4f} Global_avg:{:1.4f} Global_loss:{:2.4f}], Face[Failed:{:3d}/{:3d} Centered:{:3d}/{:3d} Moved:{:3d}/{:3d} Accuracy:{:1.2f}]".format(
           os.path.basename(video_file), answer, vid_num, len(train_video_files),
           avg_detect / num_frames, final_decision.item(),  loss_global.item(),
           n_detect_failed, num_frames, n_center_loc, num_frames, n_face_moved,
           num_frames, avg_face_accu / num_frames)
       logger.info(video_info)
       print(video_info)

       # logger.info("===> Epoch[{}](global): Loss: {:.4f}".format(epoch, loss_global.item()))
       logger.info("\n")

   #SAVE MODEL
   if opt.save:
       model_out_path = "./frame_model_epoch_{}.pth".format(epoch)
       torch.save(model, model_out_path)
       print("Checkpoint saved to {}".format(model_out_path))

       model_out_path2 = "./global_model_epoch_{}.pth".format(epoch)
       torch.save(model_global, model_out_path2)
       print("Checkpoint saved to {}".format(model_out_path2))

'''
'''
# Test
# 실제 돌릴 때 주석처리하기.
print("Testing the model...")
model.eval()
model_global.eval()

logloss = 0.0
acc = 0

for video_file in valid_video_files:

   capture = cv2.VideoCapture(video_file)

   out_tensor = torch.tensor([]).to(device)

   for iteration in range(num_frames):
       #각 File 내에서 최초 num_frames 의 frame을 각 Frame단위로 Fack Decision
       # 왜 그런지는 모르겠는데 제 컴퓨터에서는 retrieve() 가 안되고 read() 가 되어서 각자 알맞게 수정하면 될 듯 합니다 ;ㅅ;
       # 이전 코드 : ret, frame = capture.retrieve()
       ret, frame = capture.read()

       if opt.crop:
           face_loc, face = face_detector_2.crop_face(frame, face_box)
           face_box = face_loc
           frame = face
           frame = cv2.resize(frame, (h_size, v_size), interpolation=cv2.INTER_LINEAR)

       # frame = frame[300:712, 600:1112]
       frame = np.swapaxes(frame, 0, 2)
       frame = np.swapaxes(frame, 1, 2)

       frame = frame[Nol_Loss: 0.0000ne, :, :, :]
       frame = torch.from_numpy(frame).float().to(device)

       decision, fc_out = model(frame)
       frame = frame.detach()
       out_tensor = torch.cat( (out_tensor, fc_out), 0)

   #출력된 tensor를 이용하여 최종 Decision을 하는 알고리즘 개선
   global_fc_input = out_tensor[None, :, :]
   final_decision = model_global(global_fc_input)
   out_tensor = out_tensor.detach()
   global_fc_input = global_fc_input.detach()
   final_decision = final_decision.detach()

   # check a right answer
   if final_decision.item() > 0.5:
       final_out = "REAL"
   else:
       final_out = "FAKE"

   answer = train_metadata.loc[os.path.basename(video_file), 'label']

   # calculate accuracy
   if answer == final_out:
       acc += 1

   # calculate kaggle loss
   if answer == "FAKE":
       logloss += np.log(1-final_decision.item())
   else:
       logloss += np.log(final_decision.item())

   print("{} : predict {} / answer {}".format(os.path.basename(video_file), final_out, answer))

print("="*10)
print("accuracy : {}/{}".format(acc, len(valid_video_files)))
print("kaggle loss : {}".format((-1)*(logloss/len(valid_video_files))))
'''
