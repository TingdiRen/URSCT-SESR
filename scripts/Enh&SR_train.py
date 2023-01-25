import os
import time
import random
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from utils.dir_utils import mkdir, get_last_path
from utils.model_utils import network_parameters, load_checkpoint, load_start_epoch, load_best_metrics, load_optim
from utils.image_utils import torchPSNR, torchSSIM, pad_img
from dataset.data_loader import get_validation_SR_data, get_training_SR_data
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.URSCT_SR_model import URSCT_SR
from pytorch_msssim import MS_SSIM
from loss.Gradient_Loss import Gradient_Loss
from loss.Charbonnier_Loss import L1_Charbonnier_loss as Charbonnier_Loss



def main(opt):
    model_detail_opt = opt['MODEL_DETAIL']
    train_opt = opt['TRAINING']
    optim_opt = opt['OPTIM']
    ## Set Seeds
    seed = 4
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ## Path configurate
    model_name = train_opt['MODEL_NAME']
    model_dir = mkdir(os.path.join(train_opt['SAVE_DIR'], model_name, 'models'))
    result_dir = mkdir(os.path.join(train_opt['SAVE_DIR'], model_name, 'results'))
    log_dir = mkdir(os.path.join(train_opt['SAVE_DIR'], model_name, 'log'))
    train_data_dir = train_opt['TRAIN_DIR']
    val_data_dir = train_opt['VAL_DIR']

    ## Build Model
    print('================= Building Model =================')
    model = URSCT_SR(model_detail_opt).cuda()

    ## GPU
    gpus = ','.join([str(i) for i in opt['GPU']])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    ## Log Writer
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{model_name}')

    ## Optimizer
    new_lr = float(optim_opt['LR_INITIAL'])
    optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ## Learning rate strategy
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, optim_opt['EPOCHS'] - warmup_epochs,
                                                            eta_min=float(optim_opt['LR_MIN']))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    #scheduler.step()

    ## Loss
    MS_SSIM_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3).cuda()
    Gradient_loss = Gradient_Loss().cuda()
    Charbonnier_loss = Charbonnier_Loss().cuda()

    ## Resume
    start_epoch = 1
    best_psnr, best_ssim = 0., 0.
    if train_opt['RESUME']:
        print("================= Loading Resuming configuration ================= ")
        path_model_resume = get_last_path(model_dir, '_latest.pth')
        load_checkpoint(model, path_model_resume)
        start_epoch = load_start_epoch(path_model_resume) + 1
        best_psnr, best_ssim = load_best_metrics(path_model_resume)
        load_optim(optimizer, path_model_resume)
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_last_lr()[0]

    ## DataLoaders
    print("================= Creating dataloader ================= ")
    train_loader = DataLoader(dataset=get_training_SR_data(train_data_dir, {'patch_size': train_opt['TRAIN_PS']},
                                         model_detail_opt['SCALE']),
                              batch_size=optim_opt['BATCH'], shuffle=True, num_workers=train_opt['NUM_WORKS'])
    val_loader = DataLoader(dataset=get_validation_SR_data(val_data_dir, {'patch_size': train_opt['VAL_PS']},
                                                           model_detail_opt['SCALE']),
                            batch_size=1, shuffle=True, num_workers=train_opt['NUM_WORKS'])

    # Show the training configuration
    print(f'''================ Training details ================
    ------------------------------------------------------------------
        Model Name:   {model_name}
        Train patches size: {str(train_opt['TRAIN_PS']) + ' x ' + str(train_opt['TRAIN_PS'])}
        Val patches size:   {str(train_opt['VAL_PS']) + ' x ' + str(train_opt['VAL_PS'])}
        Model parameters:   {network_parameters(model)}
        Start/End epochs:   {str(start_epoch) + '~' + str(optim_opt['EPOCHS'])}
        Best PSNR:          {str(best_psnr)}
        Best SSIM:          {str(best_ssim)}
        Batch sizes:        {optim_opt['BATCH']}
        Learning rate:      {new_lr}
        GPU:                {'GPU' + str(device_ids)}''')
    print('------------------------------------------------------------------')

    # Start training
    total_start_time = time.time()
    train_show = True
    print('================ Training ================')
    for epoch in range(start_epoch, optim_opt['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss_total_SR, epoch_loss_l1_SR, epoch_loss_msssim_SR, epoch_loss_gradient_SR = 0, 0, 0, 0

        ## train
        model.train()

        pbar = tqdm(train_loader)
        for i, data in enumerate(pbar):
            optimizer.zero_grad()

            input_ = data[0].cuda()
            target_SR = data[1].cuda()
            restored_SR = model(input_)

            loss_l1_charbonnier_SR = Charbonnier_loss(restored_SR, target_SR)
            loss_msssim_SR = 1 - MS_SSIM_loss(restored_SR, target_SR)
            loss_gradient_SR = Gradient_loss(restored_SR, target_SR)
            loss_SR = loss_l1_charbonnier_SR + loss_msssim_SR + 2 * loss_gradient_SR

            if train_show:
                with torch.no_grad():
                    pbar.set_description(
                        "[Epoch] {} [Mode] TRAIN [Loss_SR] {:.4f} [PSNR_SR] {:.4f} [SSIM_SR] {:.4f}".format(
                            epoch, loss_SR.item(),
                            torchPSNR(restored_SR, target_SR),
                            torchSSIM(restored_SR,target_SR))
                    )
            # backward
            loss_SR.backward()
            optimizer.step()

            epoch_loss_l1_SR += loss_l1_charbonnier_SR.item()
            epoch_loss_gradient_SR += loss_gradient_SR.item()
            epoch_loss_msssim_SR += loss_msssim_SR.item()
            epoch_loss_total_SR += loss_SR.item()

        ## Evaluation
        if epoch % train_opt['VAL_INTERVAL'] == 0:
            model.eval()
            PSNRs, SSIMs = [], []
            pbar = tqdm(val_loader)
            for ii, data_val in enumerate(pbar, 0):
                input_ = data_val[0].cuda()
                target_SR = data_val[1].cuda()
                restored_SR = model(input_)
                with torch.no_grad():
                    for res, tar in zip(restored_SR, target_SR):
                        temp_psnr = torchPSNR(res, tar)
                        temp_ssim = torchSSIM(restored_SR, target_SR)
                        PSNRs.append(temp_psnr)
                        SSIMs.append(temp_ssim)
                        pbar.set_description("[Epoch] {} [MODE] VALID [PSNR] {:.4f} [SSIM] {:.4f}".format(
                            epoch,
                            torchPSNR(restored_SR, target_SR),
                            torchSSIM(restored_SR, target_SR))
                        )

            PSNRs = torch.stack(PSNRs).mean().item()
            SSIMs = torch.stack(SSIMs).mean().item()

            save_image(torch.cat((pad_img(input_[0], (train_opt['VAL_PS'][0], train_opt['VAL_PS'][1])),
                                  restored_SR[0], target_SR[0]), -1),
                       os.path.join(result_dir, str(epoch) + '.png'))  # save image

            # Save the best PSNR model of validation
            if PSNRs > best_psnr:
                best_psnr = PSNRs
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'PSNR': best_psnr,
                            'SSIM': best_ssim
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            print("[PSNR] {:.4f} [Best_PSNR] {:.4f} (epoch {})".format(PSNRs, best_psnr, best_epoch_psnr))

            # Save the best SSIM model of validation
            if SSIMs > best_ssim:
                best_ssim = SSIMs
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'PSNR': best_psnr,
                            'SSIM': best_ssim
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            print("[SSIM] {:.4f}  [Best_SSIM] {:.4f} (epoch {})".format(SSIMs, best_ssim, best_epoch_ssim))

            # """
            # Save each epoch
            # torch.save({'epoch': epoch,
            #             'state_dict': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'PSNR': best_psnr,
            #             'SSIM': best_ssim
            #             }, os.path.join(model_dir, "epoch{}.pth".format(epoch)))
            # """

            writer.add_scalar('val/PSNR', PSNRs, epoch)
            writer.add_scalar('val/SSIM', SSIMs, epoch)
        scheduler.step()

        print("------------------------------------------------------------------")
        print(
            "[Epoch] {} [Time] {:.4f} [Loss_SR] {:.4f} [Loss_C] {:.4f} [loss_m] {:.4f} [loss_g] {:.4f}  [Learning Rate] {:.6f}".format(
                epoch,
                time.time() - epoch_start_time,
                epoch_loss_total_SR,
                epoch_loss_l1_SR,
                epoch_loss_msssim_SR,
                epoch_loss_gradient_SR,
                scheduler.get_last_lr()[0]))
        print("------------------------------------------------------------------")

        # Save the last model
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'PSNR': best_psnr,
                    'SSIM': best_ssim
                    }, os.path.join(model_dir, "model_latest.pth"))

        writer.add_scalar('train/loss', epoch_loss_total_SR, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
    writer.close()

    print('Total training time: {:.1f} hours'.format(((time.time() - total_start_time) / 60 / 60)))


if __name__ == '__main__':
    ## Load yaml configuration file
    print('================= Loading Configuration =================')
    with open('../configs/Enh&SR_opt.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    main(opt)