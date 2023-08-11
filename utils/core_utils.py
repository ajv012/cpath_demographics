import numpy as np


import torch
from utils.utils import *
import os
# normal models
from models.model_clam import CLAM_SB
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_transmil import TransMIL

# adv models 
from models.model_clam_adv import CLAM_SB_ADV
from models.model_transmil_adv import TransMIL_ADV

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from topk.svm import SmoothTop1SVM

from torch.utils.tensorboard import SummaryWriter 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name = 'unk', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def train(datasets, fold_idx, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(fold_idx))
    writer_dir = os.path.join(args.results_dir, str(fold_idx))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    writer = SummaryWriter(writer_dir, flush_secs=15)

    print('\nInit train/val/test splits...', end=' ')
    train_loader = datasets.get('train', None)
    val_loader = datasets.get('val', None)
    test_loader = datasets.get('test', None)
    
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        loss_fn = loss_fn.cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  "in_dim": args.in_dim,
                  'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            if args.mitigation == "adv_train":
                model = CLAM_SB_ADV(**model_dict, instance_loss_fn=instance_loss_fn, n_classes_adv=args.n_classes_race)
            else:
                model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)

    elif args.model_type == "mil":
        model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes, "in_dim": args.in_dim}
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            if args.mitigation == "adv_train":
                raise NotImplementedError
            else:
                model = MIL_fc(**model_dict)

    elif args.model_type == "transmil":
        model_dict = {"n_classes": args.n_classes, "in_dim": args.in_dim}
        if args.mitigation == "adv_train":
            model = TransMIL_ADV(**model_dict, n_classes_adv=args.n_classes_race)
        else:
            model = TransMIL(**model_dict)
    
    else:
        raise NotImplementedError
    
    model.cuda()   
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopper = EarlyStopping(save_dir = args.results_dir,
                                      patience = args.es_patience, 
                                      min_stop_epoch = args.es_min_epochs,
                                      better='min', 
                                      verbose = True)
    else:
        early_stopper = None
    print('Done!')

    for epoch in range(args.max_epochs):
        print('TRAIN Epoch: ', epoch)

        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:
            if args.mitigation == "adv_train":
                loss_fn_race = nn.CrossEntropyLoss()
                train_results = train_loop_clam_adv(model, train_loader, optimizer, args.bag_weight, loss_fn, loss_fn_race, args.alpha_race,
                                                print_every=args.print_every)
        
                val_results = validate_clam_adv(model, val_loader, loss_fn, loss_fn_race, print_every=args.print_every)
            else:     
                train_results = train_loop_clam(model, train_loader, optimizer, args.bag_weight, loss_fn, 
                                                print_every=args.print_every)
                val_results = validate_clam(model, val_loader, loss_fn, print_every=args.print_every)
        
        else:
            if args.mitigation == "adv_train":
                loss_fn_race = nn.CrossEntropyLoss()
                train_results = train_loop_adv(model, train_loader, optimizer, loss_fn, loss_fn_race, args.alpha_race, print_every=args.print_every)
                val_results, _ = validate_adv(model, val_loader, loss_fn, loss_fn_race, print_every=args.print_every)
            else:
                train_results = train_loop(model, train_loader, optimizer, loss_fn, print_every=args.print_every)
                val_results, _ = validate(model, val_loader, loss_fn, print_every=args.print_every)
        
        for k,v in train_results.items():
            if k != "all_probs" and k != "all_labels": 
                writer.add_scalar(f'train/{k}', v, epoch)
        
        for k,v in val_results.items():
            if k != "all_probs" and k != "all_labels": 
                writer.add_scalar(f'val/{k}', v, epoch)

        if early_stopper is not None:
            score = val_results['cls_loss']
            save_ckpt_kwargs = dict(config = vars(args), 
                                epoch = epoch, 
                                model = model, 
                                score = score, 
                                fname = f's_{fold_idx}_checkpoint.pth')
            stop = early_stopper(epoch, score, save_checkpoint, save_ckpt_kwargs)
            if stop:
                break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pth".format(fold_idx)))['model'])
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pth".format(fold_idx)))

    val_results, val_dump = validate(model, val_loader, loss_fn, print_every=args.print_every, dump_results=True)

    print('VAL: ')
    
    for k,v in val_results.items():
        if k != "all_probs" and k != "all_labels": 
            print(f'{k}: {v:.4f}')
            writer.add_scalar(f'final/val_{k}', v, 0)
        
    writer.close()
    
    return val_dump, val_results, val_results 


def train_loop_clam(model, loader, optimizer, bag_weight, loss_fn = None, print_every = 50):
    model.train()
    acc_meter = AverageMeter()
    inst_loss_meter = AverageMeter()    
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()

    print('\n')
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        label = batch['label'].cuda()
        out = model(data, label=label, instance_eval=True)
        logits = out['logits']
        instance_loss = out['instance_loss']
        cls_loss = loss_fn(logits, label)
        acc = (label == logits.argmax(dim=-1)).float().mean()
        
        inst_loss_meter.update(instance_loss.item(), n=1)
        cls_loss_meter.update(cls_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)

        loss = bag_weight * cls_loss + (1 - bag_weight) * instance_loss 

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx}:
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_inst_loss: {inst_loss_meter.avg:.4f}, 
                    avg_acc: {acc_meter.avg:.4f},
                    avg_bag_size: {bag_size_meter.avg:.4f}""")


    return {'inst_loss': inst_loss_meter.avg, 'cls_loss': cls_loss_meter.avg, 'cls_acc': acc_meter.avg, 'bag_size': bag_size_meter.avg}

def train_loop_clam_adv(model, loader, optimizer, bag_weight, loss_fn = None, loss_fn_race = None, alpha_race=0.0001, print_every = 50):
    model.train()
    acc_meter = AverageMeter()
    inst_loss_meter = AverageMeter()    
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()

    acc_meter_race = AverageMeter()
    race_loss_meter = AverageMeter()

    print('\n')
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        label = batch['label'].cuda()
        label_race = batch["race"].cuda()

        out = model(data, label=label, instance_eval=True)

        logits = out['logits']
        logits_race = out['logits_adv']
        instance_loss = out['instance_loss']

        cls_loss = loss_fn(logits, label)
        race_loss = loss_fn_race(logits_race, label_race)

        acc = (label == logits.argmax(dim=-1)).float().mean()
        acc_race = (label_race == logits_race.argmax(dim=-1)).float().mean()
        
        inst_loss_meter.update(instance_loss.item(), n=1)
        cls_loss_meter.update(cls_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        race_loss_meter.update(race_loss.item(), n=1)
        acc_meter_race.update(acc_race.item(), n=1)

        loss = bag_weight * cls_loss + (1 - bag_weight) * instance_loss + -race_loss*alpha_race

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()

        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx}:
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_race_loss: {race_loss_meter.avg:.4f},
                    avg_inst_loss: {inst_loss_meter.avg:.4f}, 
                    avg_acc: {acc_meter.avg:.4f},
                    avg_acc_race: {acc_meter_race.avg:.4f},
                    avg_bag_size: {bag_size_meter.avg:.4f}""")


    return {'inst_loss': inst_loss_meter.avg,
     'cls_loss': cls_loss_meter.avg, 
     'race_loss': race_loss_meter.avg, 
     'cls_acc': acc_meter.avg, 
     'bag_size': bag_size_meter.avg,
     'race_acc': acc_meter_race.avg}

def train_loop(model, loader, optimizer, loss_fn = None, print_every = 50):   
    model.train()
    acc_meter = AverageMeter()
    inst_loss_meter = AverageMeter()    
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()

    print('\n')
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        
        label = batch['label'].cuda()
        out = model(data)
        logits = out['logits']
        
        cls_loss = loss_fn(logits, label)
        acc = (label == logits.argmax(dim=-1)).float().mean()
        cls_loss_meter.update(cls_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        loss = cls_loss
                
        # backward pass
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx},
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_inst_loss: {inst_loss_meter.avg:.4f},
                    avg_acc: {acc_meter.avg:.4f}, 
                    avg_bag_size: {bag_size_meter.avg:.4f}""")
        
    
    return {'cls_loss': cls_loss_meter.avg, 'cls_acc': acc_meter.avg, 'bag_size': bag_size_meter.avg}

def train_loop_adv(model, loader, optimizer, loss_fn = None, loss_fn_race = None, alpha_race=0.0001, print_every = 50):   
    model.train()
    acc_meter = AverageMeter()
    inst_loss_meter = AverageMeter()    
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()

    acc_meter_race = AverageMeter()
    race_loss_meter = AverageMeter()

    print('\n')
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        
        label = batch['label'].cuda()
        label_race = batch["race"].cuda()

        out = model(data)
        logits = out['logits']
        logits_race = out['logits_adv']
        
        cls_loss = loss_fn(logits, label)
        race_loss = loss_fn_race(logits_race, label_race)

        acc = (label == logits.argmax(dim=-1)).float().mean()
        acc_race = (label_race == logits_race.argmax(dim=-1)).float().mean()

        cls_loss_meter.update(cls_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        race_loss_meter.update(race_loss.item(), n=1)
        acc_meter_race.update(acc_race.item(), n=1)

        loss = cls_loss + -race_loss*alpha_race
                
        # backward pass
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx},
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_race_loss: {race_loss_meter.avg:.4f},
                    avg_inst_loss: {inst_loss_meter.avg:.4f},
                    avg_acc: {acc_meter.avg:.4f}, 
                    avg_acc_race: {acc_meter_race.avg:.4f},
                    avg_bag_size: {bag_size_meter.avg:.4f}""")
    
    return {'cls_loss': cls_loss_meter.avg, 
    'cls_acc': acc_meter.avg, 
    'bag_size': bag_size_meter.avg, 
    'race_loss': race_loss_meter.avg, 
    'race_acc': acc_meter_race.avg}

@torch.no_grad()  
def validate(model, loader, loss_fn = None, print_every = 50, dump_results = False, eval_mode=False):
    model.eval()
    acc_meter = AverageMeter() 
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()
    
    all_probs = []
    all_labels = []
    all_caseIDs = []
    all_races = []
    
    if eval_mode:
        insurance_dict = dict(zip(loader.dataset.data["case_id"], loader.dataset.data["insurance"]))
        age_dict = dict(zip(loader.dataset.data["case_id"], loader.dataset.data["age_group"]))
        income_dict = dict(zip(loader.dataset.data["case_id"], loader.dataset.data["income"]))
        all_age_groups = []
        all_insurance_groups = []
        all_income_groups = []
    
    
    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        label = batch['label'].cuda()
        case_id = batch["case_id"]
        race = batch["race"]
        
        out = model(data)
        
        logits = out['logits']
        cls_loss = loss_fn(logits, label)
        acc = (label == logits.argmax(dim=-1)).float().mean()
        
        cls_loss_meter.update(cls_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_caseIDs.append(case_id)
        all_races.append(race)
        
        if eval_mode:
            all_age_groups.append(age_dict[case_id[0]])
            all_income_groups.append(income_dict[case_id[0]])
            all_insurance_groups.append(insurance_dict[case_id[0]])
        
        
        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx},
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_bag_size: {bag_size_meter.avg:.4f}""")

        
        
    n_classes = logits.size(1)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_caseIDs = np.concatenate(all_caseIDs)
    all_races = np.concatenate(all_races)
    
    if eval_mode:
        all_age_groups = np.array(all_age_groups)
        all_income_groups = np.array(all_income_groups)
        all_insurance_groups = np.array(all_insurance_groups)
    
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    results = {'cls_loss': cls_loss_meter.avg, 
            'cls_acc': acc_meter.avg, 
            'bag_size': bag_size_meter.avg,
            'roc_auc': auc,
            }
    dump = {}
    if dump_results:
        
        dump['labels'] = all_labels
        dump['probs'] = all_probs
        dump['slide_ids'] = np.array(loader.dataset.get_ids(np.arange(len(loader))))
        dump['all_probs'] = all_probs
        dump['all_labels'] = all_labels
        dump["case_ids"] = all_caseIDs
        dump["all_races"] = all_races
        
        if eval_mode:
            dump["all_age_groups"] = all_age_groups
            dump["all_insurance_groups"] = all_insurance_groups
            dump["all_income_groups"] = all_income_groups
        
    return results, dump

@torch.no_grad()  
def validate_adv(model, loader, loss_fn = None, loss_fn_race = None, print_every = 50, dump_results = False, eval_mode=False):
    model.eval()
    acc_meter = AverageMeter() 
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()

    acc_meter_race = AverageMeter()
    race_loss_meter = AverageMeter()
    
    all_probs = []
    all_labels = []
    all_caseIDs = []
    all_races = []
    
    if eval_mode:
        insurance_dict = dict(zip(loader.dataset.data["case_id"], loader.dataset.data["insurance"]))
        age_dict = dict(zip(loader.dataset.data["case_id"], loader.dataset.data["age_group"]))
        income_dict = dict(zip(loader.dataset.data["case_id"], loader.dataset.data["income"]))
        all_age_groups = []
        all_insurance_groups = []
        all_income_groups = []

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        label = batch['label'].cuda()
        case_id = batch["case_id"]
        label_race = batch["race"].cuda()
        race = batch["race"]


        out = model(data)
        logits = out['logits']
        logits_race = out['logits_adv']

        cls_loss = loss_fn(logits, label)
        race_loss = loss_fn_race(logits_race, label_race)

        acc = (label == logits.argmax(dim=-1)).float().mean()
        acc_race = (label_race == logits_race.argmax(dim=-1)).float().mean()
        
        cls_loss_meter.update(cls_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        race_loss_meter.update(race_loss.item(), n=1)
        acc_meter_race.update(acc_race.item(), n=1)
        
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())
        all_caseIDs.append(case_id)
        all_races.append(race)
        
        if eval_mode:
            all_age_groups.append(age_dict[case_id[0]])
            all_income_groups.append(income_dict[case_id[0]])
            all_insurance_groups.append(insurance_dict[case_id[0]])
        
        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx},
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_race_loss: {race_loss_meter.avg:.4f},
                    avg_bag_size: {bag_size_meter.avg:.4f}""")
        
        
    n_classes = logits.size(1)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_caseIDs = np.concatenate(all_caseIDs)
    all_races = np.concatenate(all_races)
    
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    
    results = {'cls_loss': cls_loss_meter.avg, 
            'cls_acc': acc_meter.avg, 
            'bag_size': bag_size_meter.avg,
            'roc_auc': auc,
            'race_acc': acc_meter_race.avg,
            'race_loss': race_loss_meter.avg,
            }
    dump = {}
    if dump_results:
        
        dump['labels'] = all_labels
        dump['probs'] = all_probs
        dump['slide_ids'] = np.array(loader.dataset.get_ids(np.arange(len(loader))))
        dump['all_probs'] = all_probs
        dump['all_labels'] = all_labels
        dump["case_ids"] = all_caseIDs
        dump["all_races"] = all_races
        
        if eval_mode:
            dump["all_age_groups"] = all_age_groups
            dump["all_insurance_groups"] = all_insurance_groups
            dump["all_income_groups"] = all_income_groups
        
        
    return results, dump

@torch.no_grad()  
def validate_clam(model, loader, loss_fn = None, print_every = 50):
    model.eval()
    acc_meter = AverageMeter() 
    inst_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()
    
    all_probs = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        label = batch['label'].cuda()
        out = model(data, label=label, instance_eval=True)
        logits = out['logits']
        instance_loss = out['instance_loss']
        cls_loss = loss_fn(logits, label)
        acc = (label == logits.argmax(dim=-1)).float().mean()
        
        cls_loss_meter.update(cls_loss.item(), n=1)
        inst_loss_meter.update(instance_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())

        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx},
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_inst_loss: {inst_loss_meter.avg:.4f},
                    avg_acc: {acc_meter.avg:.4f},
                    avg_bag_size: {bag_size_meter.avg:.4f}""")
        
        
    n_classes = logits.size(1)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    return {'cls_loss': cls_loss_meter.avg, 
            'inst_loss': inst_loss_meter.avg,
            'cls_acc': acc_meter.avg, 
            'bag_size': bag_size_meter.avg,
            'roc_auc': auc,
            }

@torch.no_grad()  
def validate_clam_adv(model, loader, loss_fn = None, loss_fn_race = None, print_every = 50):
    model.eval()
    acc_meter = AverageMeter() 
    inst_loss_meter = AverageMeter()
    cls_loss_meter = AverageMeter()  
    bag_size_meter = AverageMeter()

    acc_meter_race = AverageMeter()
    race_loss_meter = AverageMeter()
    
    all_probs = []
    all_labels = []

    for batch_idx, batch in enumerate(loader):
        data = batch['img'].squeeze(0).cuda()
        label = batch['label'].cuda()
        label_race = batch["race"].cuda()

        out = model(data, label=label, instance_eval=True)

        logits = out['logits']
        logits_race = out['logits_adv']
        instance_loss = out['instance_loss']

        cls_loss = loss_fn(logits, label)
        race_loss = loss_fn_race(logits_race, label_race)

        acc = (label == logits.argmax(dim=-1)).float().mean()
        acc_race = (label_race == logits_race.argmax(dim=-1)).float().mean()
        
        cls_loss_meter.update(cls_loss.item(), n=1)
        inst_loss_meter.update(instance_loss.item(), n=1)
        acc_meter.update(acc.item(), n=1)
        bag_size_meter.update(len(data), n=1)
        race_loss_meter.update(race_loss.item(), n=1)
        acc_meter_race.update(acc_race.item(), n=1)
        
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        all_labels.append(label.cpu().numpy())

        if ((batch_idx + 1) % print_every  == 0) or (batch_idx == len(loader) - 1):
            print(f"""batch {batch_idx},
                    avg_cls_loss: {cls_loss_meter.avg:.4f}, 
                    avg_inst_loss: {inst_loss_meter.avg:.4f},
                    avg_race_loss: {race_loss_meter.avg:.4f},
                    avg_acc: {acc_meter.avg:.4f},
                    avg_bag_size: {bag_size_meter.avg:.4f}""")
        
        
    n_classes = logits.size(1)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    return {'cls_loss': cls_loss_meter.avg, 
            'inst_loss': inst_loss_meter.avg,
            'cls_acc': acc_meter.avg, 
            'bag_size': bag_size_meter.avg,
            'roc_auc': auc,
            'race_acc': acc_meter_race.avg,
            'race_loss': race_loss_meter.avg,
            }
