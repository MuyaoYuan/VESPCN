import torch
from trainer import Trainer
from reloader import Reloader
from implementor import Implementor
from option import args

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

if args.task == 'preparation':
    print("Selected task: preparation")
    trainer = Trainer(args=args)
    trainer.train()

elif args.task == 'train':
    print("Selected task: train")
    trainer = Trainer(args=args)
    trainer.train()

elif args.task == 'reload-pre':
    print("Selected task: reload-pre")
    reloader = Reloader(args, 'pre')
    reloader.outputs_display()
    reloader.loss_display()


elif args.task == 'reload-trained':
    print("Selected task: reload-trained")
    reloader = Reloader(args, 'trained')
    reloader.outputs_display()
    reloader.loss_display()

elif args.task == 'implement-img':
    print("Selected task: implement-img")
    implementor = Implementor(args)
    implementor.img_SR('test/image_original/Amber.jpg','test/image_SR')

elif args.task == 'implement-video':
    print("Selected task: implement-video")
    implementor = Implementor(args)
    implementor.video_SR('test/video_original/kanna10.mp4','test/video_SR')

else:
    print('Please Enter Appropriate Task Type!!!')