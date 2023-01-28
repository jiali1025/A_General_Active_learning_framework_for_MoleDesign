import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import logging
import math, random, sys
import numpy as np
import argparse
from tqdm.auto import tqdm
from poly_hgraph import *
import os
import rdkit
os.environ['CUDA_LAUNCH_BLOCKING']='1'
lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)

def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger

parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True)
parser.add_argument('--vocab', required=True)
parser.add_argument('--atom_vocab', default=common_atom_vocab)
parser.add_argument('--save_dir', required=True)
parser.add_argument('--load_model', default=None)
parser.add_argument('--load_epoch', type=int, default=-1)

parser.add_argument('--rnn_type', type=str, default='LSTM')
parser.add_argument('--hidden_size', type=int, default=250)
parser.add_argument('--embed_size', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--latent_size', type=int, default=32)
parser.add_argument('--depthT', type=int, default=15)
parser.add_argument('--depthG', type=int, default=15)
parser.add_argument('--diterT', type=int, default=1)
parser.add_argument('--diterG', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=7)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--clip_norm', type=float, default=5.0)
parser.add_argument('--beta', type=float, default=0)
parser.add_argument('--step_beta', type=float, default=0.001)
parser.add_argument('--max_beta', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--anneal_rate', type=float, default=0.9)
parser.add_argument('--print_iter', type=int, default=50)
parser.add_argument('--save_iter', type=int, default=20000)
parser.add_argument('--anneal_iter', type=int, default=25000)
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--kl_anneal_iter', type=int, default=2000)
args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)
random.seed(args.seed)

vocab = [x.strip("\r\n ").split() for x in open(args.vocab)]
vocab = [ele for ele in vocab if ele != []]
MolGraph.load_fragments([x[0] for x in vocab if eval(x[-1])])
args.vocab = PairVocab([(x,y) for x,y,_ in vocab])

model = HierVAE(args).cuda()

for param in model.parameters():
    if param.dim() == 1:
        nn.init.constant_(param, 0)
    else:
        nn.init.xavier_normal_(param)

if args.load_epoch >= 0:
    model.load_state_dict(torch.load(args.save_dir + "/model." + str(args.load_epoch)))

print("Model #Params: %dK" % (sum([x.nelement() for x in model.parameters()]) / 1000,))

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

param_norm = lambda m: math.sqrt(sum([p.norm().item() ** 2 for p in m.parameters()]))
grad_norm = lambda m: math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

total_step = 0
beta = args.beta
meters = np.zeros(6)
logger = create_logger(name='complex_vocab_train_0108', save_dir='logging_new_train_complex')
debug = logger.debug
for epoch in range(args.epoch):
    dataset = DataFolder(args.train, args.batch_size)

    for batch in tqdm(dataset):
        # print('cool')
        optimizer.zero_grad()
        # print(torch.cuda.memory_allocated())
        try:
            loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)
            # print(torch.cuda.memory_allocated())
        except Exception as e:
            print('caught an error, keep going ' + str(e))
            # torch.cuda.empty_cache()
            # model_state, optimizer_state, total_step, beta = torch.load('temp.ckpt')
            # model.load_state_dict(model_state)
            # optimizer.load_state_dict(optimizer_state)
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        optimizer.step()

        meters = meters + np.array([kl_div, loss.item(), wacc.cpu() * 100, iacc.cpu() * 100, tacc.cpu() * 100, sacc.cpu() * 100])
        total_step += 1

        # total_step += 1
        # model.zero_grad()
        # loss, kl_div, wacc, iacc, tacc, sacc = model(*batch, beta=beta)

        # loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        # optimizer.step()

        # meters = meters + np.array([kl_div, loss.item(), wacc * 100, iacc * 100, tacc * 100, sacc * 100])

        if total_step % args.print_iter == 0:
            meters /= args.print_iter
            print(
                "[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
                total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model),
                grad_norm(model)))
            msg = "[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (
            total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model),
            grad_norm(model))
            sys.stdout.flush()
            debug(msg)
            meters *= 0

        if total_step % args.save_iter == 0:
            ckpt = (model.state_dict(), optimizer.state_dict(), total_step, beta)
            torch.save(ckpt, os.path.join(args.save_dir, f"model.ckpt.{total_step}"))

        if total_step % args.anneal_iter == 0:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])

        if total_step >= args.warmup and total_step % args.kl_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)

        # if total_step % args.print_iter == 0:
        #     meters /= args.print_iter
        #     print("[%d] Beta: %.3f, KL: %.2f, loss: %.3f, Word: %.2f, %.2f, Topo: %.2f, Assm: %.2f, PNorm: %.2f, GNorm: %.2f" % (total_step, beta, meters[0], meters[1], meters[2], meters[3], meters[4], meters[5], param_norm(model), grad_norm(model)))
        #     sys.stdout.flush()
        #     meters *= 0
        #
        # if args.save_iter >= 0 and total_step % args.save_iter == 0:
        #     n_iter = total_step // args.save_iter - 1
        #     torch.save(model.state_dict(), args.save_dir + "/model." + str(n_iter))
        #     scheduler.step()
        #     print("learning rate: %.6f" % scheduler.get_lr()[0])

    # del dataset
    # if args.save_iter == -1:
    #     torch.save(model.state_dict(), args.save_dir + "/model." + str(epoch))
    #     scheduler.step()
    #     print("learning rate: %.6f" % scheduler.get_lr()[0])
