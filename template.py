def set_template(args):
    if args.template == 'ESPCN':
        args.model = 'ESPCN'
        args.dataset_name = 'DIV2K'
        args.train_path_in = "dataset/DIV2K_train_LR_bicubic_X2"
        args.train_path_label = "dataset/DIV2K_train_HR"
        args.valid_path_in = "dataset/DIV2K_valid_LR_bicubic_X2"
        args.valid_path_label = "dataset/DIV2K_valid_HR"
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
        args.epochs = 100
        args.batch_size = 1
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
    
    elif args.template == 'ESPCN_multiframe':
        args.model = 'ESPCN_multiframe'
        args.dataset_name = 'vimeo90k'
        args.dataset_path = "dataset/vimeo90k/vimeo_triplet"
        args.epochs = 30
        args.batch_size = 10
        args.lr = 1e-4
        args.n_colors = 3
        args.scale = 2
        args.n_sequence = 3
        
    else:
        print('Please Enter Appropriate Template!!!')
