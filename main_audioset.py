import torch

torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from collections import Counter
import argparse
import os
import random
import copy
import math
import numpy as np
# import datasetfactory as df
import data.datasetfactory_audioset_txt as dfs
from data.dataset_txt import MelData
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, f1_score
import models.cnn14_pann_lin as cnn14
from lib.lwlrap import calculate_per_class_lwlrap, calculate_overall_lwlrap_sklearn
import pandas as pd
from lib.torch_cka_up import cka

avg_acc = []

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # training hyperparameters
    parser.add_argument('--batch-size', type=int, default=100, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=120, help='number of training epochs')

    # incremental learning
    parser.add_argument('--new-classes', type=int, default=5, help='number of classes in new task')
    parser.add_argument('--start-classes', type=int, default=30, help='number of classes in base task')

    # optimization
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr-min', type=float, default=0.0001, help='lower end of cosine decay')
    parser.add_argument('--lr-ft', type=float, default=0.01, help='learning rate for task-2 onwards')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--cosine', action='store_true', help='use cosine learning rate')

    # root folders
    parser.add_argument('--data-root', type=str, default='./data', help='root directory of dataset')
    parser.add_argument('--output-root', type=str, default='./output', help='root directory for output')

    # save and load
    parser.add_argument('--exp-name', type=str, default='kd', help='experiment name')
    parser.add_argument('--resume', action='store_true', help='use class moco')
    parser.add_argument('--resume-path', type=str, default='./checkpoint_0.pth', )
    parser.add_argument('--save', action='store_true', help='to save checkpoint')

    # loss function
    parser.add_argument('--pow', type=float, default=0.66, help='hyperparameter of adaptive weight')
    parser.add_argument('--lamda', type=float, default=5, help='weighting of classification and distillation')
    parser.add_argument('--lamda-sd', type=float, default=10, help='weighting of classification and distillation')
    parser.add_argument('--const-lamda', action='store_true',
                        help='use constant lamda value, default: adaptive weighting')

    parser.add_argument('--w-cls', type=float, default=1.0, help='weightage of new classification loss')

    # kd loss
    parser.add_argument('--kd', action='store_true', help='use kd loss')
    parser.add_argument('--w-kd', type=float, default=1.0, help='weightage of knowledge distillation loss')
    parser.add_argument('--T', type=float, default=2, help='temperature scaling for KD')

    # cka
    parser.add_argument('--cka', action='store_true', help='use cka analysis')

    args = parser.parse_args()
    return args


def _train(model, old_model, epoch, lr, train_loader, checkPoint, posi_weight):
    # print('in')

    tolerance_cnt = 0
    step = 0
    best_acc = 0
    T = args.T

    model.cuda()
    old_model.cuda()

    print(posi_weight)
    criterion_ce = nn.BCEWithLogitsLoss(pos_weight=posi_weight)

    # reduce learning rate after first epoch (LowLR)
    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        lr = args.lr_ft

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=args.weight_decay)

    if args.cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=args.lr_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)

    if len(test_classes) // CLASS_NUM_IN_BATCH > 1:
        old_model.eval()
        num_old_classes = old_model.fc.out_features

    for epoch_index in range(1, epoch + 1):

        dist_loss = 0.0
        sum_loss = 0
        sum_dist_loss = 0
        sum_cls_new_loss = 0
        sum_cls_old_loss = 0
        sum_cls_loss = 0
        sum_feat_loss = 0

        model.train()
        old_model.eval()
        old_model.freeze_weight()
        for param_group in optimizer.param_groups:
            print('learning rate: {:.4f}'.format(param_group['lr']))

        for batch_idx, (x, target) in enumerate(train_loader):

            optimizer.zero_grad()

            # Classification Loss: New task
            x, target = x.cuda().float(), target.cuda()
            dis_label = True
            # if CLASS_NUM_IN_BATCH == 25:
            targets = target

            logits, _ = model(x)

            # if len(test_classes) // CLASS_NUM_IN_BATCH == 1:
            cls_loss_new = criterion_ce(logits[:, -CLASS_NUM_IN_BATCH:], targets)

            loss = args.w_cls * cls_loss_new
            sum_cls_new_loss += cls_loss_new.item()

            # use fixed lamda value or adaptive weighting
            if args.const_lamda:
                factor = args.lamda
            else:
                factor = ((len(test_classes) / CLASS_NUM_IN_BATCH) ** (args.pow)) * args.lamda

            # Distillation : task-2 onwards
            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:

                if args.kd:
                    with torch.no_grad():
                        dist_target, cnnfeat_old = old_model(x)
                    logits_dist, cnnfeat_new = model(x)

                    cossim = nn.CosineSimilarity(dim=cnnfeat_new.view(-1).dim() - 1)
                    feat_loss = 1 - cossim(F.normalize(cnnfeat_old.view(-1), p=2, dim=cnnfeat_old.view(-1).dim() - 1),
                                           F.normalize(cnnfeat_new.view(-1), p=2, dim=cnnfeat_new.view(-1).dim() - 1))
                    # print('feat', feat_loss)
                    sum_feat_loss += feat_loss.item()
                    loss += feat_loss

            if len(test_classes) // CLASS_NUM_IN_BATCH > 1:

                if args.kd:
                    with torch.no_grad():
                        dist_target, _ = old_model(x)
                    logits_dist = logits[:, :-CLASS_NUM_IN_BATCH]
                    T = args.T
                    dist_loss_new = nn.KLDivLoss()(F.log_softmax(logits_dist / T, dim=1),
                                                   F.softmax(dist_target / T, dim=1)) * (T * T)

                    dist_loss = dist_loss_new
                    sum_dist_loss += dist_loss.item()
                    loss += factor * args.w_kd * dist_loss

            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print(
                    '==>>> epoch: {}, batch index: {}, step: {}, train loss: {:.3f}, dist_loss: {:3f}, cls_new_loss: '
                    '{:.3f}, cls_old_loss: {:.3f}'.
                    format(epoch_index, batch_idx + 1, step, sum_loss / (batch_idx + 1),
                           sum_dist_loss / (batch_idx + 1), sum_cls_new_loss / (batch_idx + 1),
                           sum_cls_old_loss / (batch_idx + 1)))
        scheduler.step()


def evaluate_bin(model, test_loader):
    model.cuda()
    model.eval()

    all_preds = []
    all_targets = []

    for j, (mels, labels) in enumerate(test_loader):
        out, _ = model(mels.cuda().float())
        preds = torch.gt(torch.sigmoid(out), 0.5)
        all_preds.extend(
            preds.cpu().numpy())
        all_targets.extend(np.asarray(labels))

    Y_predicted = np.asarray(all_preds)
    Y_ref = np.asarray(all_targets)

    print('Reference polyphony:', Counter(Y_ref.sum(axis=1)))
    print('Predicted polyphony:', Counter(Y_predicted.sum(axis=1)))
    print(classification_report(Y_ref, Y_predicted))

    average_precision = average_precision_score(Y_ref, Y_predicted, average=None)
    mAp = np.mean(average_precision)
    print('mAP', mAp)

    f1_macro = f1_score(Y_ref, Y_predicted, average='macro')
    f1_micro = f1_score(Y_ref, Y_predicted, average='micro')
    print('macro', f1_macro)
    print('micro', f1_micro)

    Y_ref_lwrf = Y_ref
    Y_ref_lwrf[0:1, :] = 0
    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(Y_ref_lwrf, Y_predicted)
    lwlrap_value = np.sum(per_class_lwlrap * weight_per_class)
    print("lwlrap from per-class values=", lwlrap_value)

    print('on initial classes')
    print(classification_report(Y_ref[:, :30], Y_predicted[:, :30]))
    average_precision_ini = average_precision_score(Y_ref[:, :30], Y_predicted[:, :30], average=None)
    mAp_ini = np.mean(average_precision_ini)
    print('mAP_ini', mAp_ini)

    f1_macro_ini = f1_score(Y_ref[:, :30], Y_predicted[:, :30], average='macro')
    f1_micro_ini = f1_score(Y_ref[:, :30], Y_predicted[:, :30], average='micro')
    print('macro', f1_macro_ini)
    print('micro', f1_micro_ini)

    per_class_lwlrap, weight_per_class = calculate_per_class_lwlrap(Y_ref_lwrf[:, :30], Y_predicted[:, :30])
    lwlrap_value_ini = np.sum(per_class_lwlrap * weight_per_class)
    print("lwlrap from per-class values=", lwlrap_value_ini)

    # test_acc = np.mean(f1_test)

    return f1_micro, f1_macro, mAp, lwlrap_value, f1_micro_ini, f1_macro_ini, mAp_ini, lwlrap_value_ini


def cka_func(model1, model2, dataloader):
    model1.eval()
    model2.eval()
    model1.cuda()
    model2.cuda()
    cka_alg = cka.CKA(model1, model2,
                      model1_name="phase0_model",  # good idea to provide names to avoid confusion
                      model2_name="phase4_model",
                      model1_layers=['conv_block1.conv1', 'conv_block1.conv2', 'conv_block2.conv1', 'conv_block2.conv2',
                                     'conv_block3.conv1', 'conv_block3.conv2', 'conv_block4.conv1', 'conv_block4.conv2',
                                     'conv_block5.conv1', 'conv_block5.conv2', 'conv_block6.conv1', 'conv_block6.conv2',
                                     'conv_block1.bn1', 'conv_block1.bn2', 'conv_block2.bn1', 'conv_block2.bn2',
                                     'conv_block3.bn1', 'conv_block3.bn2', 'conv_block4.bn1', 'conv_block4.bn2',
                                     'conv_block5.bn1', 'conv_block5.bn2', 'conv_block6.bn1', 'conv_block6.bn2', 'fc'],
                      # List of layers to extract features from
                      model2_layers=['conv_block1.conv1', 'conv_block1.conv2', 'conv_block2.conv1', 'conv_block2.conv2',
                                     'conv_block3.conv1', 'conv_block3.conv2', 'conv_block4.conv1', 'conv_block4.conv2',
                                     'conv_block5.conv1', 'conv_block5.conv2', 'conv_block6.conv1', 'conv_block6.conv2',
                                     'conv_block1.bn1', 'conv_block1.bn2', 'conv_block2.bn1', 'conv_block2.bn2',
                                     'conv_block3.bn1', 'conv_block3.bn2', 'conv_block4.bn1', 'conv_block4.bn2',
                                     'conv_block5.bn1', 'conv_block5.bn2', 'conv_block6.bn1', 'conv_block6.bn2', 'fc'],
                      device='cuda')

    cka_alg.compare(dataloader)  # secondary dataloader is optional

    results = cka_alg.export()
    diag_sim = np.array(results['CKA'])
    #print(results)
    print('CKA scores:', np.diag(diag_sim))


if __name__ == '__main__':
    args = parse_option()
    print(args)

    if not os.path.exists(os.path.join(args.output_root, "checkpoints/audioset/")):
        os.makedirs(os.path.join(args.output_root, "checkpoints/audioset/"))

    #  parameters
    TOTAL_CLASS_NUM = 50
    CLASS_NUM_IN_BATCH = args.start_classes
    TOTAL_CLASS_BATCH_NUM = TOTAL_CLASS_NUM // CLASS_NUM_IN_BATCH
    T = args.T

    all_f1_micro = []
    all_f1_macro = []
    all_mAp = []
    all_lwlrap_value = []
    all_f1_micro_ini = []
    all_f1_macro_ini = []
    all_mAp_ini = []
    all_lwlrap_value_ini = []

    class_index = [i for i in range(0, TOTAL_CLASS_NUM)]
    # print(class_index)
    np.random.seed(1993)

    net = cnn14.Cnn14(classes_num=args.start_classes).cuda()

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('number of trainable parameters: ', params)

    old_net = copy.deepcopy(net)
    old_net.cuda()

    cls_list = [0] + [a for a in range(args.start_classes, TOTAL_CLASS_NUM, args.new_classes)]

    phase = 0

    for i in cls_list:

        if i == args.start_classes:
            CLASS_NUM_IN_BATCH = 5

        print("==> Current Class: ", class_index[i:i + CLASS_NUM_IN_BATCH])
        print('==> Building model..')

        if i == args.start_classes:
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH)
        if i > args.start_classes:
            net.change_output_dim(new_dim=i + CLASS_NUM_IN_BATCH, second_iter=True)

        print("current net output dim:", net.get_output_dim())

        train = dfs.DatasetFactory.get_dataset('audioset', train=True, phase=phase)
        val = dfs.DatasetFactory.get_dataset('audioset', train=False, phase=phase)

        train_xs = train.data
        train_ys = train.targets

        val_xs = val.data
        val_ys = val.targets

        y = torch.tensor(train_ys)
        print('y: 0s: {}, 1s: {}, nelement: {}'.format(
            (y == 0.).sum(), y.sum(), y.nelement()))
        pos_weight = (y == 0.).sum() / y.sum()

        trainLoader = torch.utils.data.DataLoader(MelData(train_xs, train_ys), batch_size=args.batch_size,
                                                  shuffle=True, num_workers=args.num_workers)
        valLoader = torch.utils.data.DataLoader(MelData(val_xs, val_ys), batch_size=args.batch_size,
                                                shuffle=False, num_workers=args.num_workers)

        train_classes = class_index[i:i + CLASS_NUM_IN_BATCH]
        test_classes = class_index[:i + CLASS_NUM_IN_BATCH]

        # train and save model
        if args.resume and (i == 0):
            #if i == 0:
            resume_path = '/scratch/project_2003370/manjunath/checkpoint_30.pth'

            net.load_state_dict(torch.load(resume_path))
            net.train()
        else:
            # print(net)
            net.train()

            _train(model=net, old_model=old_net, epoch=args.epochs, lr=args.lr,
                   train_loader=trainLoader, checkPoint=50, posi_weight=pos_weight)

        old_net = copy.deepcopy(net)
        old_net.cuda()

        if i == 0:
            base_net = copy.deepcopy(net)
            base_net.cuda()
            base_valLoader = valLoader

        if args.save:
            save_path = '/scratch/project_2003370/manjunath/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(net.state_dict(),
                       os.path.join(save_path, 'checkpoint_' + str(i + CLASS_NUM_IN_BATCH) + '.pth'))

        # Evaluation on testing set
        f1_micro, f1_macro, mAp, lwlrap_value, f1_micro_ini, f1_macro_ini, mAp_ini, lwlrap_value_ini = \
            evaluate_bin(model=net, test_loader=valLoader)

        all_f1_micro.append(f1_micro)
        all_f1_macro.append(f1_macro)
        all_mAp.append(mAp)
        all_lwlrap_value.append(lwlrap_value)
        all_f1_micro_ini.append(f1_micro_ini)
        all_f1_macro_ini.append(f1_macro_ini)
        all_mAp_ini.append(mAp_ini)
        all_lwlrap_value_ini.append(lwlrap_value_ini)
        forgetting_f1_macro = all_f1_macro_ini[0] - np.array(all_f1_macro_ini)
        forgetting_mAp = all_mAp_ini[0] - np.array(all_mAp_ini)
        forgetting_lwrap = all_lwlrap_value_ini[0] - np.array(all_lwlrap_value_ini)

        df_res = pd.DataFrame(list(zip(all_f1_micro, all_f1_macro, all_mAp, all_lwlrap_value, forgetting_f1_macro,
                                       forgetting_mAp, forgetting_lwrap)),
                              columns=['F1_m', 'F1_M', 'mAP', 'lwlrap', 'Fr_F1', 'Fr_mAP', 'Fr_lwlrap'])

        print(df_res)

        if phase == 4 and args.cka:
            cka_func(base_net, net, base_valLoader)

        phase += 1
