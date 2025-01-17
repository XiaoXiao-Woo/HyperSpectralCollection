import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-arch', type=str, default='SSRNET',
                            choices=[# these four models are used for ablation experiments
                                     'SpatCNN', 'SpecCNN',
                                     'SpatRNET', 'SpecRNET', 
                                     # the proposed method
                                     'SSRNET', 
                                     # these five models are used for comparison experiments
                                     'SSFCNN', 'ConSSFCNN', 
                                     'TFNet', 'ResTFNet', 
                                     'MSDCNN'
                                     ])

    parser.add_argument('-root', type=str, default='./data')    
    parser.add_argument('-dataset', type=str, default='harvard_x8',
                            choices=['PaviaU', 'Botswana', 'KSC', 'Urban', 'Pavia', 'IndianP', 'Washington', 'pavia'])
    parser.add_argument('--dataroot', type=str, default='F:\Data\HSI\harvard_x8')
    parser.add_argument('--scale_ratio', type=float, default=8)
    parser.add_argument('--n_bands', type=int, default=20)
    parser.add_argument('--n_select_bands', type=int, default=4)  # 5
    parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
    parser.add_argument('--testBatchSize', type=int, default=96, help='testing batch size')

    parser.add_argument('--model_path', type=str, 
                            default='./checkpoints/dataset_arch.pkl',
                            help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='F:\Data\HSI',
                            help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='F:\Data\HSI',
                            help='directory for resized images')

    # learning settingl
    parser.add_argument('--start_epochs', type=int, default=0,
                        help='end epoch for training')
    parser.add_argument('--n_epochs', type=int, default=2001,
                            help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=128)
 
    args = parser.parse_args()
    return args
 