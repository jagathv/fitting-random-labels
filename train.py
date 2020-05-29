from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim

import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt


from cifar10_data import CIFAR10RandomLabels

import cmd_args
import model_mlp, model_wideresnet


def get_data_loaders(args, shuffle_train=True):
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        if args.data_augmentation:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        kwargs = {'num_workers': 1, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(
            CIFAR10RandomLabels(root='./data', train=True, download=True,
                                transform=transform_train, num_classes=args.num_classes,
                                corrupt_prob=args.label_corrupt_prob),
            batch_size=args.batch_size, shuffle=shuffle_train, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            CIFAR10RandomLabels(root='./data', train=False,
                                transform=transform_test, num_classes=args.num_classes,
                                corrupt_prob=args.label_corrupt_prob),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader
    else:
        raise Exception('Unsupported dataset: {0}'.format(args.data))


def get_model(args):
    # create model
    if args.arch == 'wide-resnet':
        model = model_wideresnet.WideResNet(args.wrn_depth, args.num_classes,
                                            args.wrn_widen_factor,
                                            drop_rate=args.wrn_droprate)
    elif args.arch == 'mlp':
        n_units = [int(x) for x in args.mlp_spec.split('x')] # hidden dims
        n_units.append(args.num_classes)  # output dim
        n_units.insert(0, 32*32*3)        # input dim
        model = model_mlp.MLP(n_units)

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    # model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    return model


def train_model(args, model, train_loader, val_loader,
                start_epoch=None, epochs=None):
    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum)

    start_epoch = start_epoch or 0
    epochs = epochs or args.epochs


    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    prod_weight_norms = []
    norm_weight_norms = []

    update_count = 0
    print("TRAINING FOR " + str(args.itercount) + " ITERATIONS")

    for epoch in range(start_epoch, epochs):
        print("On epoch " + str(epoch))
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_losses_lst, train_accuracy_lst, prod_weight_norm_lst, norm_weight_norm_lst, val_loss_lst, val_accuracy_lst = train_epoch(train_loader, val_loader, model, criterion, optimizer, epoch, args)

        update_count += len(train_losses_lst)

        train_loss = np.concatenate((train_loss, train_losses_lst))
        train_acc = np.concatenate((train_acc, train_accuracy_lst))
        prod_weight_norms = np.concatenate((prod_weight_norms, prod_weight_norm_lst))
        norm_weight_norms = np.concatenate((norm_weight_norms, norm_weight_norm_lst))
        test_loss = np.concatenate((test_loss, val_loss_lst))
        test_acc = np.concatenate((test_acc, val_accuracy_lst))

        # evaluate on validation set
        # val_loss, val_prec1 = validate_epoch(val_loader, model, criterion, epoch, args)

        # test_loss.append(val_loss)
        # test_acc.append(val_prec1)
        #
        # if args.eval_full_trainset:
        #     tr_loss, tr_prec1 = validate_epoch(train_loader, model, criterion, epoch, args)


        # logging.info('Epoch num: %03d: Average train accuracy: %6.2f, Acc-val: %6.2f, L-val: %6.4f',
        #              epoch, val_prec1, val_loss)

        if update_count > args.itercount:
            print("EARLY STOPPING")
            break

    print("STOPPED")
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    test_loss = np.array(test_loss)
    test_acc = np.array(test_acc)
    prod_weight_norms = np.array(prod_weight_norms)
    norm_weight_norms = np.array(norm_weight_norms)

    print("Saving model and files")
    data_directory = "experiment_save/"


    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    experiment_name = "test"
    torch.save(model, os.path.join(data_directory, str(experiment_name) + "-finalmodel.pt"))
    print("Done saving model")
    np.save(os.path.join(data_directory, str(experiment_name) + "-loss.npy"), train_loss)
    np.save(os.path.join(data_directory, str(experiment_name) + "-trainaccuracy.npy"), train_acc)
    np.save(os.path.join(data_directory, str(experiment_name) + "-testloss.npy"), test_loss)
    np.save(os.path.join(data_directory, str(experiment_name) + "-testaccuracy.npy"), test_acc)
    np.save(os.path.join(data_directory, str(experiment_name) + "-prod_weight_norm.npy"), prod_weight_norms)
    np.save(os.path.join(data_directory, str(experiment_name) + "-weight_norm.npy"), norm_weight_norms)

    print("Done Saving")


def train_epoch(train_loader, val_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    train_losses_lst = []
    train_accuracy_lst = []
    prod_weight_norm_lst = []
    norm_weight_norm_lst = []
    val_loss_lst = []
    val_accuracy_lst = []



    for i, (input, target) in enumerate(train_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)



        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        val_loss, val_prec1 = validate_epoch(val_loader, model, criterion, epoch, args)

        # --- UPDATE EVERYTHING ------
        train_losses_lst.append(loss.item())
        train_accuracy_lst.append(prec1.item())

        update_weight_norms = np.array([np.linalg.norm(p.data.flatten().numpy()) for p in model.parameters()])
        norm_weight_norm = np.linalg.norm(update_weight_norms)
        norm_weight_norm_lst.append(norm_weight_norm)

        # prod weight norms
        prod_weight_norm = np.prod(update_weight_norms)
        prod_weight_norm_lst.append(prod_weight_norm)

        val_loss_lst.append(val_loss)
        val_accuracy_lst.append(val_prec1)
        # -----------------------------



        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return  train_losses_lst, train_accuracy_lst, prod_weight_norm_lst, norm_weight_norm_lst, val_loss_lst, val_accuracy_lst


def validate_epoch(val_loader, model, criterion, epoch, args):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(non_blocking=True)
        # input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    return losses.avg, top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def setup_logging(args):
    import datetime
    exp_dir = os.path.join('runs', args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
    logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print('Logging into %s...' % exp_dir)


def create_plots(data_directory, experiment_name):
    weight_norm_base = f"{experiment_name}-weight_norm"
    prod_weight_norm_base = f"{experiment_name}-prod_weight_norm"
    loss_base = f"{experiment_name}-loss"
    train_accuracy_base = f"{experiment_name}-trainaccuracy"
    test_loss_base = f"{experiment_name}-testloss"
    test_accuracy_base = f"{experiment_name}-testaccuracy"

    train_accuracy_arr = np.load(os.path.join(data_directory, f"{train_accuracy_base}.npy"))
    plt.plot(train_accuracy_arr)
    plt.title(f"{experiment_name} Train Accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Train Accuracy")
    plt.savefig(f"{train_accuracy_base}.png")
    plt.close()


    weight_norm_arr = np.load(os.path.join(data_directory, f"{weight_norm_base}.npy"))
    plt.plot(weight_norm_arr)
    plt.title(f"{experiment_name} Weight Norms")
    plt.xlabel("Updates")
    plt.ylabel("Weight Norm")
    plt.savefig(f"{weight_norm_base}.png")
    plt.close()

    prod_weight_norm_arr = np.load(os.path.join(data_directory, f"{prod_weight_norm_base}.npy"))
    plt.plot(prod_weight_norm_arr)
    plt.title(f"{experiment_name} Prod Weight Norms")
    plt.xlabel("Updates")
    plt.ylabel("Prod Weight Norm")
    plt.savefig(f"{prod_weight_norm_base}.png")
    plt.close()

    loss_arr = np.load(os.path.join(data_directory, f"{loss_base}.npy"))
    plt.plot(loss_arr)
    plt.title(f"{experiment_name} Loss")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.savefig(f"{loss_base}.png")
    plt.close()

    loss_arr = np.load(os.path.join(data_directory, f"{test_loss_base}.npy"))
    plt.plot(loss_arr)
    plt.title(f"{experiment_name} Test Loss")
    plt.xlabel("Updates")
    plt.ylabel("Test Loss")
    plt.savefig(f"{test_loss_base}.png")
    plt.close()

    test_accuracy_arr = np.load(os.path.join(data_directory, f"{test_accuracy_base}.npy"))
    plt.plot(test_accuracy_arr)
    plt.title(f"{experiment_name} Test Accuracy")
    plt.xlabel("Updates")
    plt.ylabel("Test Accuracy")
    plt.savefig(f"{test_accuracy_base}.png")
    plt.close()




def main():
    args = cmd_args.parse_args()
    setup_logging(args)

    if args.command == 'train':
        train_loader, val_loader = get_data_loaders(args, shuffle_train=True)
        model = get_model(args)
        logging.info('Number of parameters: %d', sum([p.data.nelement() for p in model.parameters()]))
        train_model(args, model, train_loader, val_loader)
        # simply create some plots of all the
        data_directory = "experiment_save/"
        experiment_name = "test"
        create_plots(data_directory, experiment_name)

if __name__ == '__main__':
    main()

