from model.rms import RMSNet
import numpy as np
import random
import torch
import torch.nn as nn
from os.path import exists
import os
from dataset.dataset import ClipDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR

GLOBAL_SEED = 7


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)


def rate(step, factor, warmup):
    if step == 0:
        step = 1
    return factor * (min(step**(-0.5), step * warmup**(-1.5)))


def run_epoch(gpu,
              data_loader,
              criterion_cls,
              criterion_reg,
              optimizer,
              scheduler,
              model,
              mode="train"):
    if mode == "train":
        model.train()
    else:
        model.eval()
    total_loss = 0
    correct_cnt = 0
    total_sample = 0
    for X, y in data_loader:
        X = X.cuda(gpu)
        y = y.cuda(gpu)
        out_cls, out_reg = model.forward(X)
        loss = criterion_cls(out_cls,
                             y[:, 0])  # + 10* criterion_reg(out_reg, y[:,1])
        y_pred = torch.argmax(out_cls, dim=1)
        correct_cnt += torch.sum(y_pred == y[:, 0]).item()
        total_sample += y.shape[0]
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item()
    return total_loss / total_sample, correct_cnt / total_sample


def train_worker(gpu, ngpu, config):
    torch.cuda.set_device(gpu)
    distributed = config['distributed']
    is_main_process = True
    if distributed:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=ngpu,
                                rank=gpu)
        is_main_process = gpu == 0

    model = RMSNet()
    model.cuda(gpu)
    module = model
    if distributed:
        model = DDP(model, device_ids=[gpu])
        module = model.module

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion_cls.cuda(gpu)
    criterion_reg.cuda(gpu)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'])

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, 0.1, config['warmup']))

    train_dataset = ClipDataset(config['data_path'], train=True)
    val_dataset = ClipDataset(config['data_path'], train=False)

    if distributed:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=ngpu,
                                           rank=gpu)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=(distributed is False),
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)
    if distributed:
        val_sampler = DistributedSampler(val_dataset,
                                         num_replicas=ngpu,
                                         rank=gpu)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=config['batch_size'],
                                             shuffle=(distributed is False),
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = 0
    for epoch in range(config['epoch_num']):
        if distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        loss, acc = run_epoch(gpu,
                              train_loader,
                              criterion_cls,
                              criterion_reg,
                              optimizer,
                              lr_scheduler,
                              model,
                              mode="train")
        if is_main_process:
            lr = optimizer.param_groups[0]["lr"]
            print((
                "TRAIN | Epoch: %6d | Loss: %6.4f | Accuracy: %6.4f | lr: %6.1e"
            ) % (epoch, loss, acc, lr))
            train_loss.append(loss)
            train_acc.append(acc)
        loss, acc = run_epoch(gpu,
                              val_loader,
                              criterion_cls,
                              criterion_reg,
                              optimizer,
                              lr_scheduler,
                              model,
                              mode="val")
        if is_main_process:
            print(("  VAL | Epoch: %6d | Loss: %6.4f | Accuracy: %6.4f") %
                  (epoch, loss, acc))
            val_loss.append(loss)
            val_acc.append(acc)
            if acc > best_val_acc:
                best_val_acc = acc
                file_path = "%sbest.pt" % config["save_prefix"]
                torch.save(module.state_dict(), file_path)

        if is_main_process:
            file_path = "%s%.2d.pt" % (config["save_prefix"], epoch)
            torch.save(module.state_dict(), file_path)

    if is_main_process:
        best_val_acc = max(val_acc)
        print("====== BEST VALSET ACCURACY: %6.4f ======" % best_val_acc)
        file_path = "%sfinal.pt" % config["save_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(config):
    ngpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train_worker, nprocs=ngpus, args=(ngpus, config))


def train_model(config):
    print("Start training.")
    if config["distributed"]:
        train_distributed_model(config)
    else:
        train_worker(0, 1, config)


def load_trained_model():
    print("Loading model...")
    config = {
        'data_path': '/home/trunk/zyx/SocDetect/data',
        'batch_size': 24,
        'epoch_num': 50,
        'save_prefix': 'weights/rms_',
        'distributed': False,
        'warmup': 5000,
        'base_lr': 0.05
    }
    model_path = "%sfinal.pt" % config["save_prefix"]
    if not exists(model_path):
        train_model(config)

    model = RMSNet()
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")
    return model


if __name__ == '__main__':
    load_trained_model()
