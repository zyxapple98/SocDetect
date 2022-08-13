from model.rms import RMSNet
import numpy as np
import random
import torch
import torch.nn as nn
from os.path import exists
import os
from dataset.dataset import ClipDataset
from torch.utils.data import random_split
import math

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


def warm_up_cosine_annealing(step, warmup, Tmax, max=1, min=1e-2):
    if step < warmup:
        return step / warmup
    else:
        return min + 0.5 * (max - min) * (1.0 + math.cos(
            (step - warmup) / (Tmax - warmup) * math.pi))


def run_epoch(gpu,
              data_loader,
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
        data_loader = tqdm(data_loader)
    for X, y in data_loader:
        X = X.cuda(gpu)
        cls_label = y[:, 0].long().cuda(gpu)
        reg_label = y[:, 1].type(torch.float32).cuda(gpu)
        out_cls, out_reg = model.forward(X)
        reg_label = torch.where(torch.isnan(reg_label), out_reg.squeeze(),
                                reg_label)
        loss = criterion_cls(out_cls, cls_label) + 10 * criterion_reg(
            out_reg.squeeze(), reg_label)
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

    model = RMSNet(backbone=config['backbone'])
    # for k, v in model.named_parameters():
    #     print('{}: {}'.format(k, v.requires_grad))
    model.cuda(gpu)
    module = model
    if distributed:
        model = DDP(model, device_ids=[gpu])
        module = model.module

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    criterion_cls.cuda(gpu)
    criterion_reg.cuda(gpu)

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=config['base_lr'],
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = LambdaLR(optimizer=optimizer,
                            lr_lambda=lambda step: warm_up_cosine_annealing(
                                step, config['warmup'], config['Tmax']))

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_acc = 0
    for epoch in range(config['epoch_num']):
        dataset = ClipDataset(config['data_path'])
        train_dataset, val_dataset = random_split(
            dataset=dataset,
            lengths=[int(len(dataset) * 4 / 5),
                     int(len(dataset) / 5)])

        if distributed:
            train_sampler = DistributedSampler(train_dataset,
                                               num_replicas=ngpu,
                                               rank=gpu)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
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
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config['batch_size'],
            shuffle=(distributed is False),
            sampler=val_sampler,
            worker_init_fn=worker_init_fn)

        if distributed:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        loss, acc, err, time = run_epoch(gpu,
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
                "TRAIN | Epoch: %6d | Loss: %6.4f | Accuracy: %6.4f | RegErr: %6.4f | Time: %6.1f | lr: %6.1e"
            ) % (epoch, loss, acc, err, time, lr))
            train_loss.append(loss)
            train_acc.append(acc)
        loss, acc, err, time = run_epoch(gpu,
                                         val_loader,
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
        'backbone': 'resnet152',
        'data_path': '/home/trunk/zyx/SocDetect/data',
        'batch_size': 24,
        'epoch_num': 50,
        'save_prefix': 'weights/rms_',
        'distributed': True,
        'warmup': 191,  # warm up during epoch 1
        'Tmax': 191 * 50,  # reach minimal lr at epoch 50
        'base_lr': 0.025
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
    #
    load_trained_model()
