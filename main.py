import argparse
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from lib.data import SyntheticDataset, DataLoaderGPU, create_if_not_exist_dataset
from lib.metrics import mean_corr_coef as mcc
from lib.models import  dagVAE
from lib.utils import Logger, checkpoint

LOG_FOLDER = 'log/'
TENSORBOARD_RUN_FOLDER = 'runs/'
TORCH_CHECKPOINT_FOLDER = 'ckpt/'



class WarmupKLLoss:

    def __init__(self, init_weights, steps,
                 M_N=0.005,
                 eta_M_N=1e-5,
                 M_N_decay_step=3000):
        """
        """
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1. - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.eta_M_N - self.M_N) / self.M_N_decay_step
        self.mn=M_N

    def _get_stage(self, step):
        while True:

            if self.stage > len(self.steps) - 1:
                break

            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.)
            elif i < stage:
                w = 1.
            else:
                w = self.init_weights[i]

            
            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.:
              
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            self.mn = min(self.M_N + self._M_N_decay_speed *
                      (step - self._ready_start_step), self.eta_M_N)
        else:
            self.mn = self.M_N

        return self.mn * loss



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default='/data/Yuhang/dagVAE-master/data/tcl_1000_40_5_5_3_1_gauss_lrelu_u_dg2.npz', help='path to data file in .npz format. (default None)')
    parser.add_argument('-x', '--data-args', type=str, default=None,
                        help='argument string to generate a dataset. '
                             'This should be of the form nps_ns_dl_dd_nl_s_p_a_u_n. '
                             'Usage explained in lib.data.create_if_not_exist_dataset. '
                             'This will overwrite the `file` argument if given. (default None). '
                             'In case of this argument and `file` argument being None, a default dataset '
                             'described in data.py will be created.')
    parser.add_argument('-b', '--batch-size', type=int, default=512, help='batch size (default 64)')
    parser.add_argument('-e', '--epochs', type=int, default=400, help='number of epochs (default 20)')
    parser.add_argument('-m', '--max-iter', type=int, default=None, help='max iters, overwrites --epochs')
    parser.add_argument('-g', '--hidden-dim', type=int, default=128, help='hidden dim of the networks (default 50)')
    parser.add_argument('-d', '--depth', type=int, default=4, help='depth (n_layers) of the networks (default 3)')
    parser.add_argument('-l', '--lr', type=float, default=5e-3, help='learning rate (default 1e-3)')
    parser.add_argument('-s', '--seed', type=int, default=1, help='random seed (default 1)')
    parser.add_argument('-c', '--cuda', action='store_true', default=True, help='train on gpu')
    parser.add_argument('-p', '--preload-gpu', action='store_true', default=True, dest='preload',
                        help='preload data on gpu for faster training.')
    parser.add_argument('-a', '--anneal', action='store_true', default=False, help='use annealing in learning')
    parser.add_argument('-n', '--no-log', action='store_true', default=True, help='run without logging')
    parser.add_argument('-q', '--log-freq', type=int, default=25, help='logging frequency (default 25).')

    parser.add_argument('-et', '--edge-threshold', type=float, default=0.1, help='threshold to cut edge.')
    parser.add_argument('-rate', '--edge-rate', type=float, default=0.7, help='rate to determine edge accoros u')


        # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.2,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.001,
                        help='The constant value used for min KL coeff')
    
    args = parser.parse_args()

    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    st = time.time()

    if args.file is None:
        args.file = create_if_not_exist_dataset(root='data/', arg_str=args.data_args)
    metadata = vars(args).copy()
    del metadata['no_log'], metadata['data_args']

    device = torch.device('cuda' if args.cuda else 'cpu')
    print('training on {}'.format(torch.cuda.get_device_name(device) if args.cuda else 'cpu'))

    # load data
    if not args.preload:
        dset = SyntheticDataset(args.file, 'cpu')
        loader_params = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = DataLoader(dset, shuffle=True, batch_size=args.batch_size, **loader_params)
        data_dim, latent_dim, aux_dim = dset.get_dims()
        args.N = len(dset)
        metadata.update(dset.get_metadata())
    else:
        train_loader = DataLoaderGPU(args.file, shuffle=True, batch_size=args.batch_size)
        data_dim, latent_dim, aux_dim = train_loader.get_dims()
        args.N = train_loader.dataset_len
        metadata.update(train_loader.get_metadata())
    if args.max_iter is None:
        args.max_iter = len(train_loader) * args.epochs

    # define model and optimizer
    model = dagVAE(latent_dim, data_dim, aux_dim, n_layers=args.depth, activation='lrelu', device=device, hidden_dim=args.hidden_dim,
                 anneal=args.anneal)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-4)

    ste = time.time()
    print('setup time: {}s'.format(ste - st))

    # setup loggers
    logger = Logger(path=LOG_FOLDER)
    exp_id = logger.get_id()
    tensorboard_run_name = TENSORBOARD_RUN_FOLDER + 'exp' + str(exp_id) + '_'.join(
        map(str, ['', args.batch_size, args.max_iter, args.lr, args.hidden_dim, args.depth, args.anneal]))
    writer = SummaryWriter(logdir=tensorboard_run_name)
    logger.add('elbo')
    logger.add('perfz')
    logger.add('perfn')
    
    print('Beginning training for exp: {}'.format(exp_id))

    # training loop
    it = 0
    model.train()
    #a = np.logspace(0,9,10,base=2)
    warmup_kl = WarmupKLLoss(init_weights=[1., 1. , 1., 1., 1.],
                             steps=[100, 100, 100, 100, 100],
                             M_N= len(train_loader)/args.batch_size,
                             eta_M_N=1.,
                             M_N_decay_step=10000)
    print('M_N=', warmup_kl.M_N, 'ETA_M_N=', warmup_kl.eta_M_N)
   #200*79 31600
    while it < args.max_iter:
        est = time.time()
       
        for _, (x, u, z, sn) in enumerate(train_loader):
            model.anneal(args.N, args.max_iter, it)
            optimizer.zero_grad()

            if args.cuda and not args.preload:
                x = x.cuda(device=device, non_blocking=True)
                u = u.cuda(device=device, non_blocking=True)

            elbo, z_est, noise_est, weight_sample, weights_u0 = model.elbo(x, u)
            #warmup_kl_loss = warmup_kl.get_loss(it, kl_all)
            #weights_all = model.WeightMat()
            elbo = elbo - 0.*torch.norm(weight_sample,1)

            #elbo = (recon + klloss).mean() - 0.*torch.norm(weight_sample,1)
            elbo.mul(-1).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            optimizer.step()

            logger.update('elbo', -elbo.item())

            if it % args.log_freq == 0:
                logger.log()
                writer.add_scalar('data/elbo', logger.get_last('elbo'), it)
                #scheduler.step() #logger.get_last('elbo')

            if it % int(args.max_iter / 5) == 0 and not args.no_log:
                checkpoint(TORCH_CHECKPOINT_FOLDER, exp_id, it, model, optimizer,
                           logger.get_last('elbo'))
            
            it += 1


        #eveluate:
        #data = np.load(path)
        model.eval()
        alldata_u = train_loader.u
        alldata_x = train_loader.x
        alldata_s = train_loader.s
        alldata_nb = train_loader.nb
        
        Allsz = np.empty((200,latent_dim))
        Allsn = np.empty((200,latent_dim))
        Allez = np.empty((200,latent_dim))
        Allen = np.empty((200,latent_dim))
        for _, (x, u, z, sn) in enumerate(train_loader):
            elbo, z_est, noise_est, weight_sample, weights_u0 = model.elbo(x, u)
           
            Allsz = np.append(Allsz, z.cpu().detach().numpy(), axis=0)
            Allsn = np.append(Allsn, sn.cpu().detach().numpy(), axis=0)
            
            Allez = np.append(Allez, z_est.cpu().detach().numpy(), axis=0)
            Allen = np.append(Allen, noise_est.cpu().detach().numpy(), axis=0)
        
        estn = Allen[200:,:]
        estz = Allez[200:,:]
        
        truez = Allsz[200:,:]
        truen = Allsn[200:,:]
            

        #elbo, z_est, noise_est, weight_sample, _ = model.elbo(alldata_x, alldata_u)

        weights_all = model.WeightMat()
        perf, assign_z = mcc(truez, estn, permute=True)
        perfn, assign_n = mcc(truen, estn, permute=True)



        
        logger.update('elbo', -elbo.item())
        logger.update('perfz', perf)
        logger.update('perfn', perfn)

        logger.log()
        writer.add_scalar('data/perf_z_n',logger.get_last('perfz'), logger.get_last('perfn'), it)

        eet = time.time()
        print('epoch {} done in: {:.3f}s, loss: {:.3f}, \t Latent_z: {:.3f}, Latent_n: {:.3f}'.format(int(it / len(train_loader)) + 1, eet - est,
                                                                   logger.get_last('elbo'), logger.get_last('perfz'), logger.get_last('perfn')))
        
        model.train()
        
        
       

    et = time.time()
    print('training time: {}s'.format(et - ste))

    writer.close()
    if not args.no_log:
        logger.add_metadata(**metadata)
        logger.save_to_json()
        logger.save_to_npz()

    print('total time: {}s'.format(et - st))
