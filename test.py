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
# import matplotlib.pyplot as plt
import re
import logging, time

from mtcnn import mtcnn


parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.01')
parser.add_argument('--lr_global', type=float, default=0.001, help='Learning Rate Decrease Ratio per a Epoch')
parser.add_argument('--cuda', action='store_true', help='use cuda?', default=True)
parser.add_argument('--sr_run', type=int, default=1, help='sr repeated No.')
parser.add_argument('--network', type=str, default='vgg16', help='Network Setlection')
parser.add_argument('--global_network', type=str, default='Conv1D', help='global network selection')
parser.add_argument('--crop', type=bool, default=True, help='feed cropped image from FaceNet')
parser.add_argument('--save', type=bool, default=False, help='Whether save trained models')
parser.add_argument('--load_f_model', type=str, default='./saved_model/frame_model_epoch_19.pth', help='frame_model')
parser.add_argument('--load_g_model', type=str, default='./saved_model/global_model_epoch_19.pth', help='global_model')
parser.add_argument('--test_dir', type=str, default='./data/valid/', help='test dir')
opt = parser.parse_args()

detector = mtcnn.MTCNN()

# hyperparameters
num_frames = 50
kernel_size = 5 # Conv2D global model
h_size = 256
v_size = 256

logging.basicConfig(
    level=logging.INFO,
    filename='log_save/'+time.strftime("%Y%m%d-%H%M%S")+'.log',
)
logger = logging.getLogger('./train.log')
logger.info(opt)


# load saved models
device = torch.device("cuda" if opt.cuda else "cpu")

# memory allocate
#if torch.cuda.device_count() > 1:
#    print("use multiple GPUs")
#    model = nn.DataParallel(model, device_ids=[0]).cuda()
#    model_global = nn.DataParallel(model_global, device_ids=[1]).cuda()

model = torch.load(opt.load_f_model)
model_global = torch.load(opt.load_g_model)

model = model.to('cuda:0')
model_global = model_global.to('cuda:1')

# DATA Loader
valid_dir = opt.test_dir
valid_video_files = [valid_dir + x for x in os.listdir(valid_dir) if x.endswith('.mp4')]
valid_metadata = pd.read_json(valid_dir + 'metadata.json')
valid_metadata = valid_metadata.T
valid_metadata.head()
valid_metadata['label'].value_counts(normalize=True)
save_dir = "./crop_face/"

# Test
print("Testing the model...")
model.eval()
model_global.eval()

logloss = 0.0
acc = 0
init_frames = 10

with torch.no_grad():
    for num_v,video_file in enumerate(valid_video_files):
        capture = cv2.VideoCapture(video_file)
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        out_tensor = torch.tensor([]).to(device)

        face_box = ((width -v_size) / 2, (height - h_size) / 2, h_size, v_size)
        iteration = 0

        avg_face_accu = 0
        n_detect_failed = 0
        n_center_loc = 0
        n_face_moved = 0
        avg_detect = 0
        n_frame = 0
        for iteration in range(0, num_frames+init_frames) :

            ret, frame = capture.read()
            if opt.crop:
                if iteration < init_frames:
                    face_loc, face, detect_acc, detect_failed, center_loc, face_moved = face_detector_1.crop_face(detector, frame, face_box, h_size, v_size, width, height, run_mode=0)
                else:
                    face_loc, face, detect_acc, detect_failed, center_loc, face_moved = face_detector_1.crop_face(detector, frame, face_box, h_size, v_size, width, height, run_mode=1)

                if face.any() == None:
                    continue

		    # # crop된 이미지 저장할때 이름 지정
            #     img_name = "{}_{}".format(re.findall(valid_dir+"(.+).mp4", video_file)[0], iteration)
            #
            #     filename = os.path.join(new_folder, img_name + "_crop" + ".png")
            #     cv2.imwrite(filename, face)

                face_box = face_loc
                frame = face
                frame = cv2.resize(frame, (h_size, v_size), interpolation=cv2.INTER_LINEAR)

            if iteration >= init_frames:
                avg_face_accu += detect_acc
                n_detect_failed += detect_failed
                n_center_loc += center_loc
                n_face_moved += face_moved

                # frame = frame[300:712, 600:1112]
                frame = np.swapaxes(frame, 0, 2)
                frame = np.swapaxes(frame, 1, 2)

                frame = frame[None, :, :, :]
                frame = torch.from_numpy(frame).float().to(device)

                decision, fc_out = model(frame)


                out_tensor = torch.cat( (out_tensor, fc_out), 0)

                avg_detect += decision.item()
                # avg_detect = int(avg_detect)

                n_frame += 1

        global_fc_input = out_tensor[None, None, :, :]
        #print(global_fc_input.shape)

        final_decision = model_global(global_fc_input.cuda(1))
        # out_tensor = out_tensor.detach()
        # global_fc_input = global_fc_input.detach()
        # final_decision = final_decision.detach()

        # check a right answer
        if final_decision.item() > 0.5:
            final_out = "REAL"
        else:
            final_out = "FAKE"

        answer = valid_metadata.loc[os.path.basename(video_file), 'label']

        print ("{:s}({}) : GlobalDecision:{:.5f} Frame_detected_avg:{:1.5f} , Face_Failed:{:4d}/{:4d} , Face_centered:{:4d}/{:4d} ,  Face_moved:{:4d}/{:4d} , Face_avg_ACCU:{:1.5f}".format(
            video_file, answer, final_decision.item(), avg_detect/n_frame, n_detect_failed, n_frame, n_center_loc, n_frame, n_face_moved, n_frame, avg_face_accu/n_frame))
        #출력된 tensor를 이용하여 최종 Decision을 하는 알고리즘 개선

        # calculate accuracy
        if answer == final_out:
            acc += 1
        #print(final_decision.item())
        # calculate kaggle loss
        if answer == "FAKE":
            logloss += np.log(1-final_decision.item())
        else:
        #kaggle loss : inf
            logloss += np.log(final_decision.item())

        # print("{} : predict {} / answer {}".format(os.path.basename(video_file), final_out, answer))

    print("="*10)
    print("accuracy : {}/{}".format(acc, len(valid_video_files)))
    print("kaggle loss : {}".format((-1)*(logloss/len(valid_video_files))))