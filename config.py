import argparse

parser = argparse.ArgumentParser(description='PET Challenge')

# data in/out and dataset
parser.add_argument('--dataset_train_path', default='./0911/enhancement/train/*/*.png',
                    help='train root path')
parser.add_argument('--dataset_test_path', default='./0911/enhancement/test/AD&CN&MCI',
                    help='test root path')
parser.add_argument("--save_dir", default='v0916_b7', help="all data dir")
parser.add_argument('--v', type=str, default='v16',
                    help='version')
#
parser.add_argument('--ColorJitter', type=float, default=0.5,
                    help='transforms.ColorJitter')
parser.add_argument('--RandomRotation', type=int, default=180,
                    help='transforms.RandomRotation')
parser.add_argument('--Resize', type=int, default=224,
                    help='transforms.Resize')
# train
parser.add_argument('--batch_size', type=list, default=16,
                    help='batch size of trainset')
parser.add_argument('--k', type=int, default=20,
                    help='Cross-validation')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='gamma')
parser.add_argument('--step_size', type=int, default=4,
                    help='step_size')
# resume
parser.add_argument('--resume', type=bool, default=False,
                    help='resume')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start_epoch')

args = parser.parse_args()
