# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

import matplotlib.pyplot as plt
# %matplotlib inline

from reader import data_loader, test_data_loader, multithread_loader
from yolov3 import YOLOv3

import functools
import argparse
from utility import add_arguments, print_arguments, check_cuda

from learning_rate import exponential_with_warmup_decay

# train.py
# 输入参数
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('learning_rate',    float, 0.00001,	 "Learning rate.")
add_arg('batch_size',	   int,   16,		"Minibatch size of all devices.")
add_arg('epoc_num',		 int,   50,	   "Epoch number.")
add_arg('anchors',		 str,   '10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326',	   "anchors.")
add_arg('ignore_thresh',	float, 0.7,	 "ignore thresh.")
add_arg('num_class',	   int,   7,		"num classes.")
add_arg('use_gpu',		  bool,  True,	  "Whether use GPU.")

args = parser.parse_args()
print_arguments(args)

learning_rate = args.learning_rate
batch_size = args.batch_size

IGNORE_THRESH = args.ignore_thresh
MAX_EPOCH = args.epoc_num
NUM_CLASSES = args.num_class
ANCHORS = [int(m) for m in args.anchors.split(",")]
# 提升点： 可以改变anchor的大小，注意训练和测试时要使用同样的anchor
# ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# IGNORE_THRESH = .7
# NUM_CLASSES = 7

TRAINDIR = './insects/train'
VALIDDIR = './insects/val'

train_parameters = {
	"insects": {
		"train_images": 1693,
		"image_shape": [3, 1344, 1344],
		"class_num": 7,
		"batch_size": 16,
		"lr": 0.001,
		"weight_file": '/home/aistudio/work/model/yolo_epoch',
		"lr_epochs": [10, 50, 100, 400],
		"lr_decay": [1, 0.5, 0.25, 0.1, 0.01],
		"ap_version": '11point',
	}
}
# 绘图
def draw_train_process(title,iters,costs,accs,label_cost,lable_acc):
    plt.clf()  # 清空画布
    plt.title(title, fontsize=24)
    plt.xlabel("epoc", fontsize=20)
    plt.ylabel("loss", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig("/home/aistudio/work/model/trian-loss.jpg")
    
# train.py
if __name__ == '__main__':
    with fluid.dygraph.guard():
        model = YOLOv3('yolov3', num_classes = NUM_CLASSES, is_train=True)
      
        # 输入参数
        dataset = 'insects'
        assert dataset in ['insects']
        
        train_parameters[dataset]['batch_size'] = args.batch_size
        train_parameters[dataset]['lr']         = args.learning_rate
        train_parameters[dataset]['epoc_num']   = args.epoc_num
    
        train_params = train_parameters[dataset]
        lr = train_params["lr"]
        
        # 预训练
        if(1):
             params_file_path = train_params['weight_file']
             model_state_dict, _ = fluid.load_dygraph(params_file_path)
             model.load_dict(model_state_dict)
             
        # 优化器
        batch_size = train_params["batch_size"]
        iters = train_params["train_images"] // batch_size
        boundaries = [i * iters  for i in train_params["lr_epochs"]]
        values = [ i * lr for i in train_params["lr_decay"]]
        
        learning_rate = fluid.layers.piecewise_decay(boundaries,values)
        # 	exponential_with_warmup_decay(learning_rate=train_params["lr"], boundaries=boundaries, values=values, warmup_iter=20, warmup_factor=0.),
        
        opt = fluid.optimizer.Momentum(
                     learning_rate= learning_rate, #提升点：可以调整学习率，或者设置学习率衰减  0.001
                     momentum=0.80,
                     regularization=fluid.regularizer.L2Decay(0.005))   # 提升点： 可以添加正则化项  L2正则化函数 

        # opt = fluid.optimizer.SGD(learning_rate=learning_rate,regularization=fluid.regularizer.L2Decay(0.005))
        # opt = fluid.optimizer.Adagrad(learning_rate=learning_rate,regularization=fluid.regularizer.L2Decay(0.005))
        # opt = fluid.optimizer.Adam(learning_rate=learning_rate,regularization=fluid.regularizer.L2Decay(0.005))
        
        # train_loader = multithread_loader(TRAINDIR, batch_size= batch_size, mode='train')
        # train_loader1 = multithread_loader(VALIDDIR, batch_size= batch_size, mode='train')
        
        
        valid_loader = multithread_loader(VALIDDIR, batch_size= batch_size, mode='valid')

        min_loss = 9999
        
        all_train_iter=0
        all_train_iters=[]
        all_train_loss=[]
        all_val_loss=[]
        
        # MAX_EPOCH = 50  # 提升点： 可以改变训练的轮数
        for epoch in range(MAX_EPOCH):
            epoch = epoch + 1
            epoch_loss = 0
            train_mean_loss = 0 
            val_mean_loss = 0
            
            if(epoch%2 == 0):
                train_loader = multithread_loader(TRAINDIR, batch_size= batch_size, mode='train')
            else:
                train_loader = multithread_loader(VALIDDIR, batch_size= batch_size, mode='train')
                
                
            for i, data in enumerate(train_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                # print(11)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)
                # print(i)
                loss.backward()
                opt.minimize(loss)
                model.clear_gradients()
                
                # if i % 20 == 0:
                #     timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                #     print('{}[TRAIN]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
                
                epoch_loss = epoch_loss + loss.numpy()

            lr = opt._global_learning_rate().numpy()#获取当前学习率
            train_mean_loss = epoch_loss/i
            timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            print('{}[TRAIN]epoch {}, iter {}, output train_mean_loss: {},learning_rate: {}'.format(timestring, epoch, i, train_mean_loss, lr))
                
            # save params of model
            if ((epoch > 1 ) and (epoch % 2 == 0) or(epoch % 5 == 0) or(epoch == MAX_EPOCH)):
                fluid.save_dygraph(model.state_dict(), '/home/aistudio/work/model/yolo_epoch{}'.format(epoch))
               
                
            # 每个epoch结束之后在验证集上进行测试
            model.eval()
            
            for i, data in enumerate(valid_loader()):
                img, gt_boxes, gt_labels, img_scale = data
                gt_scores = np.ones(gt_labels.shape).astype('float32')
                gt_scores = to_variable(gt_scores)
                img = to_variable(img)
                gt_boxes = to_variable(gt_boxes)
                gt_labels = to_variable(gt_labels)
                outputs = model(img)
                loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
                                      anchors = ANCHORS,
                                      anchor_masks = ANCHOR_MASKS,
                                      ignore_thresh=IGNORE_THRESH,
                                      use_label_smooth=False)
                # if i % 1 == 0:
                #     timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
                #     print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
                    
                val_mean_loss = val_mean_loss + loss.numpy()
                
            val_mean_loss = val_mean_loss/i
            timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
            print('{}[VALID]epoch {}, iter {}, output val_mean_loss: {}'.format(timestring, epoch, i, val_mean_loss))
            
            # 最佳验证结果
            if(min_loss > val_mean_loss):
              fluid.save_dygraph(model.state_dict(), '/home/aistudio/work/model/yolo_epoch-best')
              min_loss = val_mean_loss
              timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
              print('{}[VALID]epoch {}, iter {}, output val_min_loss: {}'.format(timestring, epoch, i, min_loss))
            
            # 绘图
            if(epoch >= 1):#首个不绘制
              all_train_iter= epoch
              all_train_iters.append(all_train_iter)
              all_train_loss.append(train_mean_loss)
              all_val_loss.append(val_mean_loss)
            
            # print(all_train_iters)
            # print(all_train_loss)
            # print(all_val_loss)
            if((epoch >1 ) and (epoch % 5 == 0) or (epoch == MAX_EPOCH)):
              print('训练epoch-{}绘图完成！'.format(epoch))
              draw_train_process("training-val",all_train_iters,all_train_loss,all_val_loss,"trainning-loss","val-loss")

            model.train()


