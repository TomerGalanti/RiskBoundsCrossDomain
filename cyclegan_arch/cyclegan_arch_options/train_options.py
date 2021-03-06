from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results_cycle on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results_cycle on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results_cycle')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_A_one_sample', type=float, default=2, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B_one_sample', type=float, default=2, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--lambda_distance_A', type=float, default=1, help='weight for distance loss (A -> B)')
        self.parser.add_argument('--lambda_distance_B', type=float, default=1, help='weight for distance loss (B -> A)')
        self.parser.add_argument('--lambda_correlation_A', type=float, default=2, help='weight for correlation loss (A -> B)')
        self.parser.add_argument('--lambda_correlation_B', type=float, default=2, help='weight for correlation loss (B -> A)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results_cycle to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--no_flip'  , action='store_true', help='if specified, do not flip the images for data argumentation')
        self.parser.add_argument('--max_items', type=int, default=300, help='maximum number of items to use for expectation and std calculation')
        self.parser.add_argument('--one_sample_index', type=int, default=0)

        self.isTrain = True
