import argparse


def define_args():
    parser = argparse.ArgumentParser(description='Configurations for WSI Training')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='maximum number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
    parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
    parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
    parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
    parser.add_argument('--split_dir', type=str, default=None, 
                        help='manually specify the set of splits to use')
    parser.add_argument('--data_source', type=str, default=None, 
                        help='manually specify the set of splits to use')
    parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
    parser.add_argument('--es_min_epochs', type=int, default=50, help='early stopping min epochs')
    parser.add_argument('--es_patience', type=int, default=20, help='early stopping min patience')
    parser.add_argument('--opt', type=str, choices = ['adamW', 'sgd'], default='adamW')
    parser.add_argument('--target_col', type=str, default='label')
    parser.add_argument('--drop_out', type=float, default=0.25, help='enabel dropout')
    parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                        help='slide-level classification loss function (default: ce)')
    parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', 'transmil'], default='clam_sb', 
                        help='type of model (default: clam_sb, clam w/ single attention branch)')
    parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
    parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
    parser.add_argument('--task', type=str)
    parser.add_argument('--mitigation', type=str, default="none")
    parser.add_argument('--alpha_race', type=float, default=0.0001, help='adversarial loss weight')
    parser.add_argument('--in_dim', default=1024, type=int, help='dim of input features')
    parser.add_argument('--print_every', default=100, type=int, help='how often to print')

    ### CLAM specific options
    parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                        help='disable instance-level clustering')
    parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default='svm',
                        help='instance-level clustering loss function')
    parser.add_argument('--subtyping', action='store_true', default=False, 
                        help='subtyping problem')
    parser.add_argument('--bag_weight', type=float, default=0.7,
                        help='clam: weight coefficient for bag-level loss (default: 0.7)')
    parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
    args = parser.parse_args()

    return args