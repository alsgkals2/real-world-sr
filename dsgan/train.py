import argparse
import os
import torch.optim as optim
import torch.utils.data
import torchvision.utils as tvutils
import data_loader as loader
import yaml
import loss
import model
import utils
from utils import set_seeds, Logger
import cv2
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='Train Downscaling Models')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--crop_size_val', default=256, type=int, help='validation images crop size')
parser.add_argument('--batch_size', default=16, type=int, help='batch size used')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers used')
parser.add_argument('--num_epochs', default=300, type=int, help='total train epoch number')
parser.add_argument('--num_decay_epochs', default=150, type=int, help='number of epochs during which lr is decayed')
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate')
parser.add_argument('--adam_beta_1', default=0.5, type=float, help='beta_1 for adam optimizer of gen and disc')
parser.add_argument('--val_interval', default=1, type=int, help='validation interval')
parser.add_argument('--val_img_interval', default=30, type=int, help='interval for saving validation images')
parser.add_argument('--save_model_interval', default=30, type=int, help='interval for saving the model')
parser.add_argument('--artifacts', default='jpeg', type=str, help='selecting different artifacts type')
parser.add_argument('--dataset', default='div2k', type=str, help='selecting different datasets')
parser.add_argument('--flips', dest='flips', action='store_true', help='if activated train images are randomly flipped')
parser.add_argument('--rotations', dest='rotations', action='store_true',
                    help='if activated train images are rotated by a random angle from {0, 90, 180, 270}')
parser.add_argument('--num_res_blocks', default=8, type=int, help='number of ResNet blocks')
parser.add_argument('--ragan', dest='ragan', action='store_true',
                    help='if activated then RaGAN is used instead of normal GAN')
parser.add_argument('--wgan', dest='wgan', action='store_true',
                    help='if activated then WGAN-GP is used instead of DCGAN')
parser.add_argument('--no_highpass', dest='highpass', action='store_false',
                    help='if activated then the highpass filter before the discriminator is omitted')
parser.add_argument('--kernel_size', default=5, type=int, help='kernel size used in transformation for discriminators')
parser.add_argument('--gaussian', dest='gaussian', action='store_true',
                    help='if activated gaussian filter is used instead of average')
parser.add_argument('--no_per_loss', dest='use_per_loss', action='store_false',
                    help='if activated no perceptual loss is used')
parser.add_argument('--lpips_rot_flip', dest='lpips_rot_flip', action='store_true',
                    help='if activated images are randomly flipped and rotated before being fed to lpips')
parser.add_argument('--disc_freq', default=1, type=int, help='number of steps until a discriminator updated is made')
parser.add_argument('--gen_freq', default=1, type=int, help='number of steps until a generator updated is made')
parser.add_argument('--w_col', default=1, type=float, help='weight of color loss')
parser.add_argument('--w_tex', default=0.005, type=float, help='weight of texture loss')
parser.add_argument('--w_per', default=0.01, type=float, help='weight of perceptual loss')
parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint model to start from')
parser.add_argument('--save_path', default=None, type=str, help='additional folder for saving the data')
parser.add_argument('--saving', dest='saving', action='store_true',
                    help='if activated the model and results are not saved')
parser.add_argument('--TESTMODE', action='store_false') #MHKIM
opt = parser.parse_args()


set_seeds()

# prepare data and DataLoaders
with open('paths.yml', 'r') as stream:
    PATHS = yaml.load(stream)
if opt.dataset == 'aim2019':
    train_set = loader.TrainDataset(PATHS['aim2019'][opt.artifacts]['source'], cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.ValDataset(PATHS['aim2019'][opt.artifacts]['valid'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=16, batch_size=1, shuffle=False)
else:
    train_set = loader.TrainDataset(PATHS[opt.dataset][opt.artifacts]['hr']['train'], cropped=True, **vars(opt))
    train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batch_size, shuffle=True)
    val_set = loader.ValDataset(PATHS[opt.dataset][opt.artifacts]['hr']['valid'],
                                lr_dir=PATHS[opt.dataset][opt.artifacts]['lr']['valid'], **vars(opt))
    val_loader = DataLoader(dataset=val_set, num_workers=16, batch_size=1, shuffle=False)

# prepare neural networks
model_g = model.Generator(n_res_blocks=opt.num_res_blocks).cuda()
print('# generator parameters:', sum(param.numel() for param in model_g.parameters()))
model_d = model.Discriminator(kernel_size=opt.kernel_size, gaussian=opt.gaussian, wgan=opt.wgan, highpass=opt.highpass).cuda()
print('# discriminator parameters:', sum(param.numel() for param in model_d.parameters()))

g_loss_module = loss.GeneratorLoss(**vars(opt))

# filters are used for generating validation images
filter_low_module = model.FilterLow(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False).cuda()
filter_high_module = model.FilterHigh(kernel_size=opt.kernel_size, gaussian=opt.gaussian, include_pad=False).cuda()
if torch.cuda.is_available():
    print('Start cuda mode')
    model_g = model_g.cuda()
    model_d = model_d.cuda()
    filter_low_module = filter_low_module.cuda()
    filter_high_module = filter_high_module.cuda()

# define optimizers
optimizer_g = optim.Adam(model_g.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
optimizer_d = optim.Adam(model_d.parameters(), lr=opt.learning_rate, betas=[opt.adam_beta_1, 0.999])
start_decay = opt.num_epochs - opt.num_decay_epochs
scheduler_rule = lambda e: 1.0 if e < start_decay else 1.0 - max(0.0, float(e - start_decay) / opt.num_decay_epochs)
scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=scheduler_rule)
scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=scheduler_rule)

# load/initialize parameters
if opt.checkpoint is not None:
    checkpoint = torch.load(opt.checkpoint)
    start_epoch = checkpoint['epoch'] + 1
    iteration = checkpoint['iteration'] + 1
    model_g.load_state_dict(checkpoint['model_g_state_dict'])
    model_d.load_state_dict(checkpoint['models_d_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
    scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
    print('Continuing training at epoch %d' % start_epoch)
else:
    start_epoch = 1
    iteration = 1

# prepare tensorboard summary
summary_path = ''
if opt.saving:
    if opt.save_path is None:
        save_path = '/' + f'bs{opt.batch_size}_crop{opt.crop_size}'
    else:
        save_path = '/' + opt.save_path
    dir_index = 0
    while os.path.isdir(f'runs_{opt.dataset}/' + save_path + '/' + str(dir_index)):
        dir_index += 1
    summary_path = f'runs_{opt.dataset}/' + save_path + '/' + str(dir_index)
        
    log = Logger()
    os.makedirs(summary_path, exist_ok=True)
    log.open(summary_path + '/log_train.txt', mode='a')
    log.write('\n')

    # writer = SummaryWriter(summary_path)
    print('Saving summary into directory ' + summary_path + '/')





# training iteration
for epoch in range(start_epoch, opt.num_epochs + 1):
    train_bar = tqdm(train_loader, desc='[%d/%d]' % (epoch, opt.num_epochs))
    model_g.train()
    model_d.train()

    for input_img, disc_img in train_bar:
        iteration += 1
        if torch.cuda.is_available():
            input_img = input_img.cuda()
            disc_img = disc_img.cuda()

        # Estimate scores of fake and real images
        fake_img = model_g(input_img)
        if opt.ragan:
            real_tex = model_d(disc_img, fake_img)
            fake_tex = model_d(fake_img, disc_img)
        else:
            real_tex = model_d(disc_img)
            fake_tex = model_d(fake_img)

        # Update Discriminator network
        if iteration % opt.disc_freq == 0:
            # calculate gradient penalty
            if opt.wgan:
                rand = torch.rand(1).item()
                sample = rand * disc_img + (1 - rand) * fake_img
                gp_tex = model_d(sample)
                gradient = torch.autograd.grad(gp_tex.mean(), sample, create_graph=True)[0]
                grad_pen = 10 * (gradient.norm() - 1) ** 2
            else:
                grad_pen = None
            # update discriminator
            model_d.zero_grad()
            d_tex_loss = loss.discriminator_loss(real_tex, fake_tex, wasserstein=opt.wgan, grad_penalties=grad_pen)
            d_tex_loss.backward(retain_graph=True)
            
            #mhkim
            """
            # optimizer_d.step() 에러나서 밑에step함 참고로, loss연산->loss2연산->optim step->optim2 steop이 맞음.
            # # save data to tensorboard
            # if opt.saving:
            #     writer.add_scalar('loss/d_tex_loss', d_tex_loss, iteration)
            #     if opt.wgan:
            #         writer.add_scalar('disc_score/gradient_penalty', grad_pen.mean().data.item(), iteration)
            """
        # Update Generator network
        if iteration % opt.gen_freq == 0:
            # update discriminator
            model_g.zero_grad()
            # print(fake_tex.shape, fake_img.shape, input_img.shape) #잘나옴 체크 torch.Size([16, 1, 128, 128]) torch.Size([16, 3, 128, 128]) torch.Size([16, 3, 128, 128])
            g_loss = g_loss_module(fake_tex, fake_img, input_img)
            assert not torch.isnan(g_loss), 'Generator loss returns NaN values'
            # print(g_loss)
            g_loss.backward()

            #optim d step
            optimizer_d.step() #mhkim 위에꺼주석하고 여기로 옮김
            if opt.saving:
                log.write(f'loss/d_tex_loss : {d_tex_loss}, {iteration} \n')
                if opt.wgan:
                    temp = grad_pen.mean().data.item()
                    log.write(f'disc_score/gradient_penalty : {temp}, {iteration} \n')

            #optim g step
            optimizer_g.step()
            # save data to tensorboard
            if opt.saving:
                log.write(f'loss/perceptual_loss {g_loss_module.last_per_loss}, {iteration} \n')
                log.write(f'loss/color_loss {g_loss_module.last_col_loss}, {iteration} \n')
                log.write(f'loss/g_tex_loss {g_loss_module.last_tex_loss}, {iteration} \n')
                log.write(f'loss/g_overall_loss  {g_loss}, {iteration} \n')

        # save data to tensorboard
        rgb_loss = g_loss_module.rgb_loss(fake_img, input_img)
        mean_loss = g_loss_module.mean_loss(fake_img, input_img)
        if opt.saving:
            log.write(f'loss/rgb_loss {rgb_loss}, {iteration} \n')
            log.write(f'loss/mean_loss {mean_loss}, {iteration} \n')
            temp = real_tex.mean().data.item()
            log.write(f'disc_score/real {temp}, {iteration} \n')
            temp = fake_tex.mean().data.item(),
            log.write(f'disc_score/fake {temp}, {iteration} \n')
        train_bar.set_description(desc='[%d/%d]' % (epoch, opt.num_epochs))

        # break
    scheduler_d.step()
    scheduler_g.step()
    if opt.saving:
        temp = torch.Tensor(scheduler_g.get_lr())
        log.write(f'param/learning_rate {temp}, {epoch} \n')

    # validation step
    if epoch % opt.val_interval == 0 or epoch % opt.val_img_interval == 0:
        val_bar = tqdm(val_loader, desc='[Validation]')
        model_g.eval()
        val_images = []
        with torch.no_grad():
            # initialize variables to estimate averages
            mse_sum = psnr_sum = rgb_loss_sum = mean_loss_sum = 0
            per_loss_sum = col_loss_sum = tex_loss_sum = 0

            # validate on each image in the val dataset
            for index, (input_img, disc_img, target_img) in enumerate(val_bar):
                if torch.cuda.is_available():
                    input_img = input_img.cuda()
                    target_img = target_img.cuda()
                fake_img = torch.clamp(model_g(input_img), min=0, max=1)
                mse = ((fake_img - target_img) ** 2).mean().data
                mse_sum += mse
                psnr_sum += -10 * torch.log10(mse)
                rgb_loss_sum += g_loss_module.rgb_loss(fake_img, target_img)
                mean_loss_sum += g_loss_module.mean_loss(fake_img, target_img)
                # per_loss_sum += g_loss_module.perceptual_loss(fake_img, target_img)
                # print(fake_img.shape, target_img.shape)
                temp =  g_loss_module.perceptual_loss(fake_img, target_img)
                per_loss_sum += g_loss_module.perceptual_loss(fake_img, target_img).squeeze()
                col_loss_sum += g_loss_module.color_loss(fake_img, target_img)

                # generate images
                if epoch % 1 == 0 and epoch != 0:
                    blur = filter_low_module(fake_img)
                    hf = filter_high_module(fake_img)
                    val_image_list = [
                        utils.display_transform()(target_img.data.cpu().squeeze(0)),
                        utils.display_transform()(fake_img.data.cpu().squeeze(0)),
                        utils.display_transform()(disc_img.squeeze(0)),
                        utils.display_transform()(blur.data.cpu().squeeze(0)),
                        utils.display_transform()(hf.data.cpu().squeeze(0))]
                    n_val_images = len(val_image_list)
                    val_images.extend(val_image_list)

            if len(val_loader) > 0:
                # save validation values
                log.write(f'val/mse, {mse_sum/len(val_set)}, {iteration} \n')
                log.write(f'val/psnr, {psnr_sum / len(val_set)}, {iteration} \n')
                log.write(f'val/rgb_error, {rgb_loss_sum / len(val_set)}, {iteration} \n')
                log.write(f'val/mean_error, {mean_loss_sum / len(val_set)}, {iteration} \n')
                per_loss_sum = per_loss_sum.mean(dim=0,keepdim=False) #mhkim
                log.write(f'val/perceptual_error, {per_loss_sum / len(val_set)}, {iteration} \n')
                log.write(f'val/color_error, {col_loss_sum / len(val_set)}, {iteration} \n')

                # save image results
                if epoch % opt.val_img_interval == 0 and epoch != 0:
                    val_images = torch.stack(val_images)
                    val_images = torch.chunk(val_images, val_images.size(0) // (n_val_images * 5))
                    val_save_bar = tqdm(val_images, desc='[Saving results]')
                    for index, image in enumerate(val_save_bar):
                        image = tvutils.make_grid(image, nrow=n_val_images, padding=5)
                        image = image.squeeze()
                        image = np.array(image.permute(1,2,0)*255.)
                        savepath = f'{summary_path}/val/epoch_{epoch}'
                        os.makedirs(savepath,exist_ok=True) #mhkim
                        out_path = f'{savepath}/target_fake_tex_disc_f-wav_t-wav_' + str(index)
                        cv2.imwrite(f'{savepath}/target_fake_crop_low_high_' + str(index)+'.jpg', image)
                        # writer.add_image('val/target_fake_crop_low_high_' + str(index), image, iteration)

        # save model parameters
        # if epoch % opt.save_model_interval == 0 and epoch != 0:
        path = './checkpoints/' + save_path + '/epoch_' + str(epoch) + '.tar'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        state_dict = {
            'epoch': epoch,
            'iteration': iteration,
            'model_g_state_dict': model_g.state_dict(),
            'models_d_state_dict': model_d.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'scheduler_g_state_dict': scheduler_g.state_dict(),
            'scheduler_d_state_dict': scheduler_d.state_dict(),
        }
        torch.save(state_dict, path)
    path = './checkpoints' + save_path + '/last_iteration.tar'
    torch.save(state_dict, path)
