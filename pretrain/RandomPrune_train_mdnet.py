import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np

import torch

sys.path.insert(0,'.')
from data_prov import RegionDataset
from datasets.data import cifar10
from modules.model_random_prune import VGG_RandomPrune, VGG_Vital, set_optimizer, BCELoss, Precision

device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

#load pre-train params
def load_vgg_random_model(model):
    #print(ckpt['state_dict'])
    origin_model = VGG_RandomPrune(prune_rate=1).to(device)
    ckpt = torch.load(args.honey_model, map_location=device)
    origin_model.load_state_dict(ckpt['state_dict'])
    oristate_dict = origin_model.state_dict()

    state_dict = model.state_dict()
    last_select_index = None #Conv index selected in the previous layer

    for name, module in model.named_modules():

        if isinstance(module, nn.Conv2d):

            oriweight = oristate_dict[name + '.weight']
            curweight = state_dict[name + '.weight']
            orifilter_num = oriweight.size(0)
            currentfilter_num = curweight.size(0)

            if orifilter_num != currentfilter_num and (random_rule == 'random_pretrain' or random_rule == 'l1_pretrain'):

                select_num = currentfilter_num
                if random_rule == 'random_pretrain':
                    select_index = random.sample(range(0, orifilter_num-1), select_num)
                    select_index.sort()
                else:
                    l1_sum = list(torch.sum(torch.abs(oriweight), [1, 2, 3]))
                    select_index = list(map(l1_sum.index, heapq.nlargest(currentfilter_num, l1_sum)))
                    select_index.sort()
                if last_select_index is not None:
                    for index_i, i in enumerate(select_index):
                        for index_j, j in enumerate(last_select_index):
                            state_dict[name + '.weight'][index_i][index_j] = \
                                oristate_dict[name + '.weight'][i][j]
                else:
                    for index_i, i in enumerate(select_index):
                        state_dict[name + '.weight'][index_i] = \
                            oristate_dict[name + '.weight'][i]

                last_select_index = select_index

            else:
                state_dict[name + '.weight'] = oriweight
                last_select_index = None

    model.load_state_dict(state_dict)

# Training
def train(model, optimizer, trainLoader, opts, epoch):

    model.train()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        optimizer.step()




#Testinga
def test(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter()
    accurary = utils.AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

    return accurary.avg


def train_mdnet(opts):

    loader = cifar10.Data(args)
    best_acc = 0.0

    model_vgg = VGG_RandomPrune(opts['prune_rate']).to(device)
    load_vgg_random_model(model_vgg)


    optimizer = optim.SGD(model.parameters(), lr=opts['lr_finetune'], momentum=opts['momentum'], weight_decay=opts['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts['lr_decay_step'], gamma=0.1)

    for epoch in range(opts['num_epochs']):
        train(model, optimizer, loader.trainLoader, opts, epoch)
        scheduler.step()
        test_acc = test(model, loader.testLoader)

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

    
    model_state_dict = model.module.state_dict() if len(opts[gpus]) > 1 else model.state_dict()
    torch.save(model_state_dict, opts['pruned_model_path'])
    print('Best accurary: {:.3f}'.format(float(best_acc)))



    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    dataset = [None] * K
    for k, seq in enumerate(data.values()):
        #print(seq)
        dataset[k] = RegionDataset(seq['images'], seq['gt'], opts)

    # Init model
    model = VGG_Vital(opts['pruned_model_path'], K, opts['prune_rate'])
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])

    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)

        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K)
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next()
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            loss = criterion(pos_score, neg_score)

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))

        print('Mean Precision: {:.3f}'.format(prec.mean()))
        print('Save model to {:s}'.format(opts['model_path']))
        if opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='imagenet', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open('pretrain/options_{}.yaml'.format(args.dataset), 'r'))
    
    train_mdnet(opts)
