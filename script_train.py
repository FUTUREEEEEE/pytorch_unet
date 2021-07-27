import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model_unet import *
from datasets import CityscapesDataset

import os 
import time
import torch
from S2R import DSAModules
import numpy as np
from S2R.utils import *
import torch.nn as nn
from S2R.models import modules
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from matplotlib.image import imsave

#####
#load Shared_Struct_Encoder
Shared_Struct_Encoder = modules.Struct_Encoder(n_downsample=2, n_res=4, 
											input_dim=3, dim=64, 
											norm='in', activ='lrelu', 
											pad_type='reflect')
Shared_Struct_Encoder = Shared_Struct_Encoder.cuda()	
Shared_Struct_Encoder.load_state_dict(torch.load("/content/drive/MyDrive/S2Rnet/struct_encoder_vkitti.pth"))
#for name,p in Shared_Struct_Encoder.named_parameters():
#  print(name,p)
Shared_Struct_Encoder.eval()

#load decoder
Struct_Decoder = modules.Struct_Decoder()
Struct_Decoder = torch.nn.DataParallel(Struct_Decoder).cuda()
Struct_Decoder.load_state_dict(torch.load("/content/drive/MyDrive/S2Rnet/struct_decoder_vkitti.pth"))
Struct_Decoder.eval()

#load attention module
Attention_Model = DSAModules.drn_d_22(pretrained=True)
DSAModle = DSAModules.AutoED(Attention_Model)
DSAModle = torch.nn.DataParallel(DSAModle).cuda()
DSAModle.load_state_dict(torch.load("/content/drive/MyDrive/S2Rnet/dsamodels_vkitti.pth"))
DSAModle.eval()

#process 


########

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument("--datadir", type=str, default="/home/patrick/datasets/cityscapes/", help="directory the Cityscapes dataset is in")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
parser.add_argument("--losstype", type=str, default="segment", help="choose between segment & reconstruction")
args = parser.parse_args()


# hyper-parameters (learning rate and how many epochs we will train for)
lr = 0.0002
epochs = 100


# cityscapes dataset loading
img_data = CityscapesDataset(args.datadir, split='train', mode='fine', augment=False)
img_batch = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
print(img_data)


# loss function
# use reconstruction of image if looking to match image output to another image (RGB)
# else if you have a set of classes, we want to do some binary classification on it (cityscapes classes)
if args.losstype == "reconstruction":
    recon_loss_func = nn.MSELoss()
    num_classes = 3  # red, blue, green
elif args.losstype == "segment":
    recon_loss_func = nn.CrossEntropyLoss()
    num_classes = img_data.num_classes  # background, road, sky, car
else:
    print("please select a valid loss type (reconstruction or segment)...")
    exit()


# initiate generator and optimizer
print("creating unet model...")
generator = nn.DataParallel(UnetGenerator(3, img_data.num_classes, 64), device_ids=[i for i in range(args.num_gpu)]).cuda()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


# load pretrained model if it is there
file_model = './unet.pkl'
if os.path.isfile(file_model):
    generator = torch.load(file_model)
    print("    - model restored from file....")
    print("    - filename = %s" % file_model)


# or log file that has the output of our loss
file_loss = open('./unet_loss', 'w')


# make the result directory
if not os.path.exists('./result/'):
    os.makedirs('./result/')


# finally!!! the training loop!!!
for epoch in range(epochs):
    for idx_batch, (imagergb, labelmask, labelrgb) in enumerate(img_batch):

        # zero the grad of the network before feed-forward
        gen_optimizer.zero_grad()

        # send to the GPU and do a forward pass
        x = Variable(imagergb).cuda(0)
        with torch.no_grad():

          struct_code = Shared_Struct_Encoder(x)
          struct_code = Struct_Decoder(struct_code)

          attention_map = DSAModle(x)
          attention_map = attention_map * struct_code

          x=torch.cat((x,attention_map),1)
          
          del struct_code,attention_map

        #print("\n imagergb: ",imagergb.shape,"\n")
        y_ = Variable(labelmask).cuda(0)
        y = generator.forward(x)

        # we "squeeze" the groundtruth if we are using cross-entropy loss
        # this is because it expects to have a [N, W, H] image where the values
        # in the 2D image correspond to the class that that pixel should be 0 < pix[u,v] < classes
        if args.losstype == "segment":
            y_ = torch.squeeze(y_)

        # finally calculate the loss and back propagate
        loss = recon_loss_func(y, y_)
        file_loss.write(str(loss.item())+"\n")
        loss.backward()
        gen_optimizer.step()

        # every 400 images, save the current images
        # also checkpoint the model to disk
        if idx_batch % 400 == 0:

            # nice debug print of this epoch and its loss
            print("epoch = "+str(epoch)+" | loss = "+str(loss.item()))

            # save the original image and label batches to file
            #v_utils.save_image(x.cpu().data, "./result/original_image_{}_{}.png".format(epoch, idx_batch))
            #v_utils.save_image(labelrgb, "./result/label_image_{}_{}.png".format(epoch, idx_batch))

            # max over the classes should be the prediction
            # our prediction is [N, classes, W, H]
            # so we max over the second dimension and take the max response
            # if we are doing rgb reconstruction, then just directly save it to file
            if args.losstype == "segment":
                y_threshed = torch.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
                for idx in range(0, y.size()[0]):
                    maxindex = torch.argmax(y[idx], dim=0).cpu().int()
                    y_threshed[idx] = img_data.class_to_rgb(maxindex)
                v_utils.save_image(y_threshed, "./result/gen_image_{}_{}.png".format(epoch, idx_batch))
            else:
                v_utils.save_image(y.cpu().data, "./result/gen_image_{}_{}.png".format(epoch, idx_batch))

            # finally checkpoint this file to disk
            torch.save(generator, file_model)
