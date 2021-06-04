import torch
from torchnet import meter
from torch.utils.data import DataLoader
from Architectures.unet import CELT_Unet
import Architectures.encoders as encoders
from data.CamVid import *
import segmentation_models_pytorch as smp
import os


if __name__ == '__main__':
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    #data_dir = "/home/sang/data/CamVid/"
    data_dir = "e:/data/CamVid/"
    DEVICE = 'cuda:0'
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist','unlabelled']
    ACTIVATION = 'softmax2d'  # could be None for logits or 'softmax2d' for multicalss segmentation

    best_model_location = "./checkpoints/camvid/resnet/"
    if not os.path.exists(best_model_location):
        os.makedirs(best_model_location)

    best_model_location_and_name = best_model_location + "CELT_Unet_CamVid.pth"

    ##
    # 1. 加载数据
    ##
    ##
    # 配置
    ##
    ################################
    CamVidtrans = CamVidTransform()
    preprocessing_fn = encoders.get_preprocessing_fn(ENCODER,ENCODER_WEIGHTS)

    train_dataset = CamVid_Dataset(
        data_dir + "train/",
        data_dir+"trainannot/",
        augmentation=CamVidtrans.get_training_augmentation(),
        preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    valid_dataset = CamVid_Dataset(
        data_dir + "val/",
        data_dir + "valannot/",
        augmentation=CamVidtrans.get_validation_augmentation(),
        preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    # create test dataset
    test_dataset = CamVid_Dataset(
        data_dir + "test/",
        data_dir + "testannot/",
        augmentation=CamVidtrans.get_validation_augmentation(),
        preprocessing=CamVidtrans.get_preprocessing(preprocessing_fn),
        classes=CLASSES,
    )

    batch_size=4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_dataset,batch_size=1, shuffle=False, num_workers=1)

    ##
    # 2. 训练模型
    ##

    # 2.1加载训练模型
    if os.path.exists(best_model_location_and_name):
        model = torch.load(best_model_location_and_name)
    else:
        # create segmentation model with pretrained encoder
        model = CELT_Unet(
            encoder_name=ENCODER,
            #encoder_weights=ENCODER_WEIGHTS,
            encoder_weights=None,
            classes=len(CLASSES),
            activation=ACTIVATION
        )
    model.to(DEVICE)

    # 2.2 开始训练
    lr = 0.0001
    optimizer =torch.optim.Adam(model.parameters(), lr=lr)
    loss = smp.utils.losses.DiceLoss()

    metrics = smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Fscore(), smp.utils.metrics.Accuracy(), smp.utils.metrics.Precision(), smp.utils.metrics.Recall()

    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )
    best_iou=0
    '''
    log = test_epoch.run(test_dataloader)
    best_iou=log["iou_score"]
    print("-------------------------测试结果--------------------------")
    print("%10s\t%10s\t%10s\t%10s\t%10s\t%10s" % ("IoU score","dice loss","fscore","recall","precision","accuracy"))
    print("%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" % (
        log["iou_score"], log["dice_loss"], log["fscore"], log["recall"], log["precision"], log["accuracy"]))
    '''
    for epoch in range(500):
        print("Epoch %d ..." %epoch)
        # train model
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        if best_iou < valid_logs['iou_score']:
            best_iou = valid_logs['iou_score']
            torch.save(model, best_model_location_and_name)

        if epoch == 400:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')

        if epoch == 450:
            optimizer.param_groups[0]['lr'] = 1e-6
            print('Decrease decoder learning rate to 1e-6!')

        best_iou_cpu = torch.tensor(best_iou, device='cpu').tolist()
        # print(val_accuracy.is_cuda)
        print("%10s\t%10s\t%10s\t%10s\t%10s\t%10s" % ("IoU score", "dice loss", "fscore", "recall", "precision", "accuracy"))

        log=train_logs
        print("%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" % (log["iou_score"], log["dice_loss"], log["fscore"], log["recall"], log["precision"], log["accuracy"]))

        log=valid_logs
        print("%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f\t%10.3f" % (log["iou_score"], log["dice_loss"], log["fscore"], log["recall"], log["precision"], log["accuracy"]))

    best_model= torch.load(best_model_location_and_name)
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    test_logs = test_epoch.run(test_dataloader)
    log = test_logs
    print("-------------------------测试结果--------------------------")
    print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (
    log["iou_score"], log["dice_loss"], log["fscore"], log["recall"], log["precision"], log["accuracy"]))
