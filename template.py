def set_template(args):
    if args.template == 'ESPCN':
        args.model = 'ESPCN'
        args.dataset_name = 'DIV2K'
        args.train_path_in = "dataset/DIV2K_train_LR_bicubic_X2"
        args.train_path_label = "dataset/DIV2K_train_HR"
        args.valid_path_in = "dataset/DIV2K_valid_LR_bicubic_X2"
        args.valid_path_label = "dataset/DIV2K_valid_HR"
        args.transform = 'withTranspose'
        args.epochs = 50
        args.batch_size = 1
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2

    elif args.template == 'ESPCN_modified':
        args.model = 'ESPCN_modified'
        args.dataset_name = 'DIV2K'
        args.train_path_in = "dataset/DIV2K_train_LR_bicubic_X2"
        args.train_path_label = "dataset/DIV2K_train_HR"
        args.valid_path_in = "dataset/DIV2K_valid_LR_bicubic_X2"
        args.valid_path_label = "dataset/DIV2K_valid_HR"
        args.transform = 'null'
        args.epochs = 100
        args.batch_size = 1
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
    
    elif args.template == 'ESPCN_multiframe':
        args.model = 'ESPCN_multiframe'
        args.dataset_name = 'vimeo90k'
        args.dataset_path = "dataset/vimeo90k/vimeo_triplet"
        args.transform = 'null'
        args.epochs = 30
        args.batch_size = 10
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
        args.n_sequence = 3
    
    elif args.template == 'ESPCN_multiframe2':
        args.model = 'ESPCN_multiframe2'
        args.dataset_name = 'vimeo90k'
        args.dataset_path = "dataset/vimeo90k/vimeo_triplet"
        args.transform = 'null'
        args.epochs = 30
        args.batch_size = 10
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
        args.n_sequence = 3
    
    elif args.template == 'MC':
        args.model = 'MC'
        args.dataset_name = 'vimeo90k'
        args.dataset_path = "dataset/vimeo90k/vimeo_triplet"
        args.transform = 'null'
        args.epochs = 30
        args.batch_size = 10
        args.lr = 1e-4
        args.n_colors = 3
        args.lamda = 0.0005
    
    elif args.template == 'VESPCN':
        args.model = 'VESPCN'
        args.dataset_name = 'vimeo90k'
        args.dataset_path = "dataset/vimeo90k/vimeo_triplet"
        args.transform = 'null'
        args.epochs = 30
        args.batch_size = 10
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
        args.n_sequence = 3
        # 损失函数权重
        args.lamda = 0.0005
        args.beta = 0.005

    elif args.template == 'SRCNN':
        args.model = 'SRCNN'
        args.dataset_name = 'DIV2K'
        args.train_path_in = "dataset/DIV2K_train_HR_bicubic"
        args.train_path_label = "dataset/DIV2K_train_HR"
        args.valid_path_in = "dataset/DIV2K_valid_HR_bicubic"
        args.valid_path_label = "dataset/DIV2K_valid_HR"
        args.transform = 'withTranspose'
        args.epochs = 50
        args.batch_size = 1
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
        
    else:
        print('Please Enter Appropriate Template!!!')
