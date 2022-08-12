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
import time
from tqdm import tqdm

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
              data_loader_ev,
              data_loader_ba,
              criterion_cls,
              criterion_reg,
              optimizer,
              scheduler,
              model,
              mode="train"):
    start = time.time()
    if mode == "train":
        model.train()
    else:
        model.eval()
    total_loss = 0
    correct_cnt = 0
    reg_err = 0
    total_sample = 0
    if gpu == 0:
        data_loader_ba = tqdm(data_loader_ba)
    for X, y in data_loader_ba:
        X = X.cuda(gpu)
        cls_label = y[:, 0].long().cuda(gpu)
        # zero_label = torch.zeros(y.shape[0]).type(torch.float32).cuda(gpu)
        out_cls, out_reg = model.forward(X)
        loss = criterion_cls(out_cls.squeeze(), cls_label) + 0 * torch.mean(out_reg.squeeze())
        y_pred = torch.argmax(out_cls, dim=1)
        correct_cnt += torch.sum(y_pred == cls_label).item()
        total_sample += cls_label.shape[0]
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item()
    if gpu == 0:
        data_loader_ev = tqdm(data_loader_ev)
    for X, y in data_loader_ev:
        X = X.cuda(gpu)
        cls_label = y[:, 0].long().cuda(gpu)
        reg_label = y[:, 1].type(torch.float32).cuda(gpu)
        out_cls, out_reg = model.forward(X)
        loss = criterion_cls(
            out_cls.squeeze(),
            cls_label) + 10 * criterion_reg(out_reg.squeeze(), reg_label)
        y_pred = torch.argmax(out_cls, dim=1)
        correct_cnt += torch.sum(y_pred == cls_label).item()
        reg_err += torch.sum(torch.abs(out_reg.squeeze() - reg_label)).item()
        total_sample += cls_label.shape[0]
        if mode == "train":
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        total_loss += loss.item()
    return total_loss / total_sample, correct_cnt / total_sample, reg_err / total_sample, (
        time.time() - start) / 60


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

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=config['base_lr'])
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, 0.1, config['warmup']))

    train_dataset_ev = ClipDataset(config['data_path'],
                                   train=True,
                                   background=False)
    train_dataset_ba = ClipDataset(config['data_path'],
                                   train=True,
                                   background=True)
    val_dataset_ev = ClipDataset(config['data_path'],
                                 train=False,
                                 background=False)
    val_dataset_ba = ClipDataset(config['data_path'],
                                 train=False,
                                 background=True)

    if distributed:
        train_sampler_ev = DistributedSampler(train_dataset_ev,
                                              num_replicas=ngpu,
                                              rank=gpu)
        train_sampler_ba = DistributedSampler(train_dataset_ba,
                                              num_replicas=ngpu,
                                              rank=gpu)
    else:
        train_sampler_ev = None
        train_sampler_ba = None
    train_loader_ev = torch.utils.data.DataLoader(
        dataset=train_dataset_ev,
        batch_size=config['batch_size'],
        shuffle=(distributed is False),
        sampler=train_sampler_ev,
        worker_init_fn=worker_init_fn)
    train_loader_ba = torch.utils.data.DataLoader(
        dataset=train_dataset_ba,
        batch_size=config['batch_size'],
        shuffle=(distributed is False),
        sampler=train_sampler_ba,
        worker_init_fn=worker_init_fn)
    if distributed:
        val_sampler_ev = DistributedSampler(val_dataset_ev,
                                            num_replicas=ngpu,
                                            rank=gpu)
        val_sampler_ba = DistributedSampler(val_dataset_ba,
                                            num_replicas=ngpu,
                                            rank=gpu)
    else:
        val_sampler_ev = None
        val_sampler_ba = None
    val_loader_ev = torch.utils.data.DataLoader(
        dataset=val_dataset_ev,
        batch_size=config['batch_size'],
        shuffle=(distributed is False),
        sampler=val_sampler_ev,
        worker_init_fn=worker_init_fn)
    val_loader_ba = torch.utils.data.DataLoader(
        dataset=val_dataset_ba,
        batch_size=config['batch_size'],
        shuffle=(distributed is False),
        sampler=val_sampler_ba,
        worker_init_fn=worker_init_fn)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = 0
    for epoch in range(config['epoch_num']):
        if distributed:
            train_loader_ev.sampler.set_epoch(epoch)
            train_loader_ba.sampler.set_epoch(epoch)
            val_loader_ev.sampler.set_epoch(epoch)
            val_loader_ba.sampler.set_epoch(epoch)
        loss, acc, err, time = run_epoch(gpu,
                                         train_loader_ev,
                                         train_loader_ba,
                                         criterion_cls,
                                         criterion_reg,
                                         optimizer,
                                         lr_scheduler,
                                         model,
                                         mode="train")
        if is_main_process:
            lr = optimizer.param_groups[0]["lr"]
            print((
                "TRAIN | Epoch: %6d | Loss: %6.4f | Accuracy: %6.4f | RegErr: %6.4f | Time: %6.1f | lr: %6.1e"
            ) % (epoch, loss, acc, err, time, lr))
            train_loss.append(loss)
            train_acc.append(acc)
        loss, acc, err, time = run_epoch(gpu,
                                         val_loader_ev,
                                         val_loader_ba,
                                         criterion_cls,
                                         criterion_reg,
                                         optimizer,
                                         lr_scheduler,
                                         model,
                                         mode="val")
        module = model.module
        if is_main_process:
            print((
                "  VAL | Epoch: %6d | Loss: %6.4f | Accuracy: %6.4f | RegErr: %6.4f | Time: %6.1f"
            ) % (epoch, loss, acc, err, time))
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
        'epoch_num': 100,
        'save_prefix': 'weights/rms_',
        'distributed': True,
        'warmup': 500,
        'base_lr': 0.1
    }
    model_path = "%sbest.pt" % config["save_prefix"]
    if not exists(model_path):
        train_model(config)

    model = RMSNet()
    model.load_state_dict(torch.load(model_path))
    print("Model loaded.")
    return model


if __name__ == '__main__':
    # print(torch.__version__)
    # print(torch.cuda.is_available())
    # print(torch.version.cuda)
    load_trained_model()
