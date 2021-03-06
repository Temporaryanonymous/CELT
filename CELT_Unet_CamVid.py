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
    # Your data directory
    data_dir = "image_samples/CamVid/"

    DEVICE = 'cuda:0'
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement',
               'tree', 'signsymbol', 'fence', 'car',
               'pedestrian', 'bicyclist','unlabelled']
    ACTIVATION = 'softmax2d'  
    best_model_location = "./checkpoints/camvid/resnet/"
    if not os.path.exists(best_model_location):
        os.makedirs(best_model_location)

    best_model_location_and_name = best_model_location + "CELT_Unet_CamVid.pth"

    ##
    # 1. Loading data
    ##
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
    # 2. Train models
    ##

    # 2.1 Loading the trained model if it exists
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

    # 2.2 start training
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
    for epoch in range(500):
        print("Epoch %d ..." %epoch)
        # train model
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        if best_iou < valid_logs['iou_score']:
            best_iou = valid_logs['iou_score']
            torch.save(model, best_model_location_and_name)

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
    print("-------------------------Testing Results--------------------------")
    print("%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" % (
    log["iou_score"], log["dice_loss"], log["fscore"], log["recall"], log["precision"], log["accuracy"]))
