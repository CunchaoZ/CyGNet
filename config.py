import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS18')
args.add_argument('--time-stamp', type=int, default=15)
args.add_argument('--lr', type=float, default=0.001)
args.add_argument('--n-epochs', type=int, default=30)
args.add_argument('--hidden-dim', type=int, default=200)
args.add_argument("--gpu", type=int, default=-1,
                  help="gpu")
args.add_argument("--regularization", type=float, default=0.01,
                    help="regularization weight")
args.add_argument('--valid-epoch', type=int, default=5)
args.add_argument('--alpha', type=float, default=0.8)
args.add_argument('--batch-size', type=int, default=1024)
args.add_argument('--raw', action='store_true', default=False)
args.add_argument('--counts', type=int, default=5)

args = args.parse_args()
print(args)
