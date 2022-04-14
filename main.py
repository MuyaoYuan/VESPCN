import torch
from trainer import Trainer
from trainer_mc import Trainer_MC
from trainer_vespcn import Trainer_VESPCN
from reloader import Reloader
from implementor import Implementor
from option import args

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if args.task == 'preparation':
    print("Selected task: preparation")
    if(args.model != 'MC' and args.model != 'VESPCN'):
        trainer = Trainer(args=args)
    elif(args.model == 'MC'):
        trainer = Trainer_MC(args=args)
    elif(args.model == 'VESPCN'):
        trainer = Trainer_VESPCN(args=args)
    trainer.train()

elif args.task == 'train':
    print("Selected task: train")
    if(args.model != 'MC' and args.model != 'VESPCN'):
        trainer = Trainer(args=args)
    elif(args.model == 'MC'):
        trainer = Trainer_MC(args=args)
    elif(args.model == 'VESPCN'):
        trainer = Trainer_VESPCN(args=args)
    trainer.train()

elif args.task == 'reload-pre':
    print("Selected task: reload-pre")
    reloader = Reloader(args, 'pre')
    if(args.model != 'MC' and args.model != 'VESPCN'):
        reloader.outputs_display()
    elif(args.model == 'MC'):
        reloader.outputs_display_MC()
    elif(args.model == 'VESPCN'):
        reloader.outputs_display_VESPCN()
    reloader.loss_display()
        

elif args.task == 'reload-trained':
    print("Selected task: reload-trained")
    reloader = Reloader(args, 'trained')
    if(args.model != 'MC' and args.model != 'VESPCN'):
        reloader.outputs_display()
    elif(args.model == 'MC'):
        reloader.outputs_display_MC()
    elif(args.model == 'VESPCN'):
        reloader.outputs_display_VESPCN()
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