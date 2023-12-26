import os
import cv2
import time
import datetime
import numpy as np
import torch.autograd
from skimage import io
from scipy import stats
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.loss import CrossEntropyLoss2d
from utils.utils import accuracy, intersectionAndUnion, AverageMeter, CaclTP


# Choose model and data
##################################################
import data_loader.BLU as BLU
# import data_loader.GID as GID
# import data_loader.PD as PD
from models.TCNet import TCNet as Net
NET_NAME = 'TCNet'
DATA_NAME = 'BLU'
##################################################

# Change testing parameters here
working_path = os.path.abspath('.')
args = {
    'gpu': True,
    's_class': 0,
    'val_batch_size': 1,
    'size_local': 256,
    'size_context': 256 * 3,
    'data_dir': '/home/vcl/zhangli/data/try/BLU/',
    'load_path': '/home/vcl/zhangli/code/TCNet/checkpoints/BLU/TCNet_1e_OA59.28.pth'
}


def norm_gray(x, out_range=(0, 255)):
    # x=x*(x>0)
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0] + 1e-10)
    y = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return y.astype('uint8')


def draw_rectangle(img, pos='boundary', color=(0, 255, 0), thick=2, text='context window'):
    h, w, c = img.shape
    if pos == 'boundary':
        start_pos = (0, 0)
        end_pos = (h - 1, w - 1)
    elif pos == 'center':
        start_pos = (h // 2 - 64, w // 2 - 64)
        end_pos = (h // 2 + 64, w // 2 + 64)
    cv2.rectangle(img, start_pos, end_pos, color, thick)
    if pos == 'boundary':
        cv2.putText(img, text, (start_pos[0] + 15, start_pos[1] + 15), cv2.FONT_HERSHEY_PLAIN, 1.2, color)
    elif pos == 'center':
        cv2.putText(img, text, (start_pos[0] - 24, start_pos[1] - 5), cv2.FONT_HERSHEY_PLAIN, 1.2, color)
    return img


def main():
    net = Net(3, num_classes=BLU.num_classes + 1, size_context=args['size_context'], size_local=args['size_local']).cuda()
    # net = Net(4, num_classes=GID.num_classes+1, size_context=args['size_context'], size_local=args['size_local']).cuda()
    # net = Net(5, num_classes=PD.num_classes+1, size_context=args['size_context'], size_local=args['size_local']).cuda()

    net.load_state_dict(torch.load(args['load_path']))  # , strict = False
    net = net.cuda()
    net.eval()
    print(NET_NAME+' Model loaded.')
    pred_path = os.path.join(args['data_dir'], 'Eval', NET_NAME)
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')

    val_set = BLU.Loader(args['data_dir'], 'test', sliding_crop=True, size_context=args['size_context'], size_local=args['size_local'])
    # val_set = GID.Loader(args['data_dir'], 'test', sliding_crop=True, size_context=args['size_context'], size_local=args['size_local'])
    # val_set = PD.Loader(args['data_dir'], 'test', sliding_crop=True, size_context=args['size_context'], size_local=args['size_local'])
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
    predict(net, val_loader, pred_path, f)
    f.close()

def save_img(score_save_path,i,img_s, label_s, img, label,output,aux):
    # fm_path = score_save_path
    fm_path = os.path.join(score_save_path,'%d'%i)
    if not os.path.exists(fm_path): os.makedirs(fm_path)

    img_name = os.path.join(fm_path, '%d_img.png'%i)
    lbl_name = os.path.join(fm_path, '%d_label.png'%i)
    img_s_name = os.path.join(fm_path, '%d_img_s.png'%i)
    lbl_s_name = os.path.join(fm_path, '%d_label_s.png'%i)
    aux_name = os.path.join(fm_path, '%d_aux.png'%i)
    pred_name = os.path.join(fm_path, '%d_output.png'%i)

    img_s = img_s.detach().cpu().numpy().squeeze().transpose((1, 2, 0))[:,:,:3]
    img_s = norm_gray(img_s)
    img_s = cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB)
    label_s = label_s.detach().cpu().numpy()
    labels_color = BLU.Index2Color(label_s).squeeze()

    img = img.detach().cpu().numpy().squeeze().transpose((1, 2, 0))[:,:,:3]
    img = norm_gray(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = label.detach().cpu().numpy()
    label_color = BLU.Index2Color(label).squeeze()

    output = output.detach().cpu()
    pred = torch.argmax(output, dim=1)
    pred = pred.numpy().squeeze()
    pred_color = BLU.Index2Color(pred)

    aux = aux.detach().cpu()
    aux = torch.argmax(aux, dim=1)
    aux = aux.numpy().squeeze()
    aux_color = BLU.Index2Color(aux)

    io.imsave(img_name, img)
    io.imsave(lbl_name, label_color)
    io.imsave(img_s_name, img_s)
    io.imsave(lbl_s_name, labels_color)
    io.imsave(aux_name, aux_color)
    io.imsave(pred_name, pred_color)

def predict(net, pred_loader, pred_path, f_out=None):
    acc_meter = AverageMeter()
    TP_meter = AverageMeter()
    pred_meter = AverageMeter()
    label_meter = AverageMeter()
    Union_meter = AverageMeter()
    output_info = f_out is not None
    save_path = 'result1'
    if not os.path.exists(save_path): os.makedirs(save_path)
    for vi, data in enumerate(pred_loader):
        with torch.no_grad():
            img_s, label_s,_,_, img, label = data
            if args['gpu']:
                img_s = img_s.cuda().float()
                img = img.cuda().float()
                label = label.cuda().float()

            output, _,_,_ = net(img_s, img)  #

        output = output.detach().cpu()
        pred = torch.argmax(output, dim=1)
        pred = pred.numpy().squeeze()
        if args['s_class']:
            class_map = F.softmax(output, dim=1)[:, args['s_class'], :, :]
            class_map = class_map.numpy().squeeze()
            class_map = norm_gray(class_map)

        label = label.detach().cpu().numpy()
        acc, _ = accuracy(pred, label)
        acc_meter.update(acc)
        pred_color = BLU.Index2Color(pred)
        img = img.detach().cpu().numpy().squeeze().transpose((1, 2, 0))
        img = norm_gray(img)

        pred_name = os.path.join(pred_path, '%d_WiC.png' % vi)

        io.imsave(pred_name, pred_color)

        if args['s_class']:
            saliency_map = cv2.applyColorMap(class_map, cv2.COLORMAP_JET)
            pred_name = os.path.join(pred_path, '%d_saliency.png' % vi)
            saliency_map = (img * 0.5 + saliency_map * 0.5).astype('uint8')
            io.imsave(pred_name, saliency_map)
        TP, pred_hist, label_hist, union_hist = CaclTP(pred, label, BLU.num_classes)
        TP_meter.update(TP)
        pred_meter.update(pred_hist)
        label_meter.update(label_hist)
        Union_meter.update(union_hist)
        print('Eval num %d/%d, Acc %.2f' % (vi, len(pred_loader), acc * 100))
        if output_info:
            f_out.write('Eval num %d/%d, Acc %.2f\n' % (vi, len(pred_loader), acc * 100))

    precision = TP_meter.sum / (label_meter.sum + 1e-10) + 1e-10
    recall = TP_meter.sum / (pred_meter.sum + 1e-10) + 1e-10
    F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    F1 = np.array(F1)
    IoU = TP_meter.sum / Union_meter.sum
    IoU = np.array(IoU)

    print(output.shape)
    print('Acc %.2f' % (acc_meter.avg * 100))
    avg_F = F1.mean()
    mIoU = IoU.mean()
    print('Avg F1 %.2f' % (avg_F * 100))
    print(np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    print('mIoU %.2f' % (mIoU * 100))
    print(np.array2string(IoU * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    if output_info:
        f_out.write('Acc %.2f\n' % (acc_meter.avg * 100))
        f_out.write('Avg F1 %.2f\n' % (avg_F * 100))
        f_out.write(
            np.array2string(F1 * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
        f_out.write('\nmIoU %.2f\n' % (mIoU * 100))
        f_out.write(
            np.array2string(IoU * 100, precision=4, separator=', ', formatter={'float_kind': lambda x: "%.2f" % x}))
    return avg_F


if __name__ == '__main__':
    main()