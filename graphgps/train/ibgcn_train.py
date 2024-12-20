import torch
import numpy as np
import time
import logging

from torch.nn import functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, \
    clean_ckpt

from torch_geometric.graphgym.register import register_train


def vib_loss(y_pred, y, mean, std):
    cross_entropy = F.cross_entropy(y_pred, y,reduction='sum')
    kl = 0.5*torch.mean(mean.pow(2)+std.pow(2)- 2*std.log()-1)
    beta = 1e-3
    
    return beta*kl+cross_entropy, y_pred 

def vib_train_epoch(logger, loader, model, optimizer, scheduler, batch_accumulation):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    k = torch.randint(1,model.max_hops,(1,))[0]
    for iter, batch in enumerate(loader):
        batch.split = 'train'
        batch.to(torch.device(cfg.device))
        
        output, y_true  = model(batch, k)
        y_pred, mu, std = output
        loss, pred_score = vib_loss(y_pred,y_true,mu,std)
        #loss, pred_score = compute_loss(output, y_true)
        loss.backward()
        if((iter+1)%batch_accumulation == 0) or (iter +1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        logger.update_stats(true=y_true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()

@torch.no_grad()
def eval_epoch(logger, loader, model, split='val'):
    model.eval()
    time_start = time.time()
    for batch in loader:
        batch.split = split
        batch.to(torch.device(cfg.device))
        output, y_true = model(batch, model.max_hops)
        #loss, pred_score = compute_loss(output, y_true)
        y_pred, mu, std = output
        loss, pred_score = vib_loss(y_pred,y_true,mu,std)
        logger.update_stats(true=y_true.detach().cpu(),
                            pred=pred_score.detach().cpu(), loss=loss.item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()


def vib_train(loggers, loaders, model, optimizer, scheduler):
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler)
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    split_names = ['val','test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        vib_train_epoch(loggers[0], loaders[0], model, optimizer, scheduler,
                        cfg.optim.batch_accumulation)
        perf[0].append(loggers[0].write_epoch(cur_epoch))
        #loggers[0].write_epoch(cur_epoch)

        if is_eval_epoch(cur_epoch):
            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i-1])
                delme = loggers[i].write_epoch(cur_epoch)
                perf[i].append(delme)
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter()-start_time)
        if is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)
        
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"

                else: 
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                "\n\n"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                "\n\n"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
                "\n\n"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")


    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    logging.info('Task done, results saved in {}'.format(cfg.run_dir))


register_train('vib_train', vib_train)
