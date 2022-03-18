from models.sync_batchnorm import DataParallelWithCallback
import models.generator as generators
import models.discriminator as discriminators
import os
import copy
import torch
from torch.nn import init
import models.losses as losses
from models.CannyFilter import CannyFilter
from abc import ABC, abstractmethod
from torch import nn, autograd, optim


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


#Abstract Model Class 
class Model(nn.Module,ABC):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        # --- generator and discriminator ---
        if opt.netG == 1 :
            self.netG = generators.wavelet_generator(opt)
        elif opt.netG == 2 :
            self.netG = generators.wavelet_generator_multiple_levels(opt)
        elif opt.netG == 3 :
            self.netG = generators.wavelet_generator_multiple_levels_no_tanh(opt)
        elif opt.netG == 4:
            self.netG = generators.IWT_spade_upsample_WT_generator(opt)
        elif opt.netG == 5:
            self.netG = generators.wavelet_generator_multiple_levels_reductive_upsample(opt)
        elif opt.netG == 6:
            self.netG = generators.IWT_spade_upsample_WT_reductive_upsample_generator(opt)
        elif opt.netG == 7:
            self.netG = generators.progGrow_Generator(opt)
        elif opt.netG == 8:
            self.netG = generators.ResidualWaveletGenerator(opt)
        elif opt.netG == 9:
            self.netG = generators.ResidualWaveletGenerator_1(opt)
        elif opt.netG == 10:
            self.netG = generators.ResidualWaveletGenerator_2(opt)
        else :
            self.netG = generators.OASIS_Generator(opt)

        if opt.phase == "train":
            self.netD = discriminators.OASIS_Discriminator(opt)
            if opt.netDu == 'wavelet':
                self.netDu = discriminators.WaveletDiscriminator(opt)
            else :
                self.netDu = discriminators.TileStyleGAN2Discriminator(3, opt=opt)
            self.criterionGAN = losses.GANLoss("nonsaturating")
            self.featmatch = torch.nn.MSELoss()
        self.print_parameter_count()
        self.init_networks()
        # --- EMA of generator weights ---
        with torch.no_grad():
            self.netEMA = copy.deepcopy(self.netG) if not opt.no_EMA else None
        # --- load previous checkpoints if needed ---
        self.load_checkpoints()
        # --- perceptual loss ---#
        if opt.add_edges :
            self.canny_filter = CannyFilter(use_cuda= (self.opt.gpu_ids != -1) )
        if opt.phase == "train":
            if opt.add_vgg_loss:
                self.VGG_loss = losses.VGGLoss(self.opt.gpu_ids)
            if opt.add_edge_loss:
                self.BDCN_loss = losses.BDCNLoss(self.opt.gpu_ids)
    @abstractmethod
    def forward(self):
        pass

    def compute_edges(self,images):
        if self.opt.add_edges :
            edges = self.canny_filter(images,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
        else :
            edges = None

        return edges

    def load_checkpoints(self):
        if self.opt.phase == "test":
            which_iter = self.opt.ckpt_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            if self.opt.no_EMA:
                self.netG.load_state_dict(torch.load(path + "G.pth"))
            else:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))
        elif self.opt.continue_train:
            which_iter = self.opt.which_iter
            path = os.path.join(self.opt.checkpoints_dir, self.opt.name, "models", str(which_iter) + "_")
            self.netG.load_state_dict(torch.load(path + "G.pth"))
            self.netD.load_state_dict(torch.load(path + "D.pth"))
            self.netDu.load_state_dict(torch.load(path + "Du.pth"))

            if not self.opt.no_EMA:
                self.netEMA.load_state_dict(torch.load(path + "EMA.pth"))

    def print_parameter_count(self):
        if self.opt.phase == "train":
            networks = [self.netG, self.netD, self.netDu]
        else:
            networks = [self.netG]
        for network in networks:
            print('Created', network.__class__.__name__,
                  "with %d parameters" % sum(p.numel() for p in network.parameters()))

    def init_networks(self):
        def init_weights(m, gain=0.02):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                #if not (m.weight.data.shape[0] == 3 and m.weight.data.shape[2] == 1 and m.weight.data.shape[3] == 1) :
                    init.xavier_normal_(m.weight.data, gain=gain)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)

        if self.opt.phase == "train":
            networks = [self.netG, self.netD,]
        else:
            networks = [self.netG]
        for net in networks:
            net.apply(init_weights)


class Unpaired_model(Model):
    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        if self.opt.add_edges :
            edges = self.canny_filter(image,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
            import matplotlib.pyplot as plt
            plt.imshow(edges.cpu()[0, 0, ...])
            plt.show()
        else :
            edges = None

        if mode == "losses_G_unsupervised":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            output_D = self.netD(fake)
            loss_G_adv = self.opt.lambda_segment*losses_computer.loss(output_D, label, for_real=True)

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

            pred_fake = self.netDu(fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
            loss_G += loss_G_GAN

            if self.opt.add_edge_loss:
                loss_G_edge = self.opt.lambda_edge * self.BDCN_loss(label, fake )
                loss_G += loss_G_edge
            else:
                loss_G_edge = None

            return loss_G, [loss_G_adv, loss_G_vgg, loss_G_GAN, loss_G_edge]


        if mode == "losses_G_supervised":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            fake_features = self.netDu(fake,for_features = True)
            real_features = self.netDu(image,for_features = True)

            loss_G_feat = 0
            for real_feat,fake_feat in zip(real_features,fake_features):
                loss_G_feat += self.featmatch(real_feat,fake_feat)

            loss_G += loss_G_feat

            return loss_G,[loss_G_feat]

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

        if mode == "losses_D":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=True)
            loss_D += loss_D_fake
            output_D_real = output_D_fake
            loss_D_real = None

            """if self.opt.model_supervision == 2 :
                output_D_real = self.netD(image)
                loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
                loss_D += loss_D_real"""

            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
                                                                                     output_D_fake,
                                                                                     output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "losses_Du":
            loss_Du = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            output_Du_fake = self.netDu(fake)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            loss_Du += loss_Du_fake

            output_Du_real = self.netDu(image)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()
            loss_Du += loss_Du_real

            return loss_Du, [loss_Du_fake,loss_Du_real]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake

        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regularize":
            loss_Du = 0
            image.requires_grad = True
            real_pred = self.netDu(image)
            r1_loss = d_r1_loss(real_pred, image).mean()
            loss_Du += 10 * r1_loss
            return loss_Du, [r1_loss]


class Semi_supervised_model(Model):    
    def forward(self, image, label, mode, losses_computer):
        # Branching is applied to be compatible with DataParallel
        if self.opt.add_edges :
            edges = self.canny_filter(image,low_threshold = 0.1,high_threshold = 0.3,hysteresis = True)[-1].detach().float()
            import matplotlib.pyplot as plt
            plt.imshow(edges.cpu()[0, 0, ...])
            plt.show()
        else :
            edges = None

        if mode == "losses_G_unsupervised":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            output_D = self.netD(fake)
            loss_G_adv = self.opt.lambda_segment*losses_computer.loss(output_D, label, for_real=True)

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

            pred_fake = self.netDu(fake)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean()
            loss_G += loss_G_GAN

            if self.opt.add_edge_loss:
                loss_G_edge = self.opt.lambda_edge * self.BDCN_loss(label, fake )
                loss_G += loss_G_edge
            else:
                loss_G_edge = None

            return loss_G, [loss_G_adv, loss_G_vgg, loss_G_GAN, loss_G_edge]


        if mode == "losses_G_supervised":
            loss_G = 0
            fake = self.netG(label,edges = edges)
            fake_features = self.netD(fake)
            loss_G_adv = self.opt.lambda_segment*losses_computer.loss(fake_features, label, for_real=True)

            # loss_G_adv = torch.zeros_like(loss_G_adv)
            loss_G += loss_G_adv
            if self.opt.add_vgg_loss:
                loss_G_vgg = self.opt.lambda_vgg * self.VGG_loss(fake, image)
                loss_G += loss_G_vgg
            else:
                loss_G_vgg = None

            return loss_G, [loss_G_adv, loss_G_vgg]

        # if mode == "losses_D_unsupervised":
        #     loss_D = 0
        #     with torch.no_grad():
        #         fake = self.netG(label,edges = edges)
        #     output_D_fake = self.netD(fake)
        #     loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=True)
        #     loss_D += loss_D_fake
        #     output_D_real = output_D_fake
        #     loss_D_real = None

        #     """if self.opt.model_supervision == 2 :
        #         output_D_real = self.netD(image)
        #         loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
        #         loss_D += loss_D_real"""

        #     if not self.opt.no_labelmix:
        #         mixed_inp, mask = generate_labelmix(label, fake, image)
        #         output_D_mixed = self.netD(mixed_inp)
        #         loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed,
        #                                                                              output_D_fake,
        #                                                                              output_D_real)
        #         loss_D += loss_D_lm
        #     else:
        #         loss_D_lm = None
        #     return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]
            
        if mode == "losses_D_supervised":
            loss_D = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            output_D_fake = self.netD(fake)
            loss_D_fake = losses_computer.loss(output_D_fake, label, for_real=False)
            loss_D += loss_D_fake
            output_D_real = self.netD(image)
            loss_D_real = losses_computer.loss(output_D_real, label, for_real=True)
            loss_D += loss_D_real
            if not self.opt.no_labelmix:
                mixed_inp, mask = generate_labelmix(label, fake, image)
                output_D_mixed = self.netD(mixed_inp)
                loss_D_lm = self.opt.lambda_labelmix * losses_computer.loss_labelmix(mask, output_D_mixed, output_D_fake,
                                                                                output_D_real)
                loss_D += loss_D_lm
            else:
                loss_D_lm = None
            return loss_D, [loss_D_fake, loss_D_real, loss_D_lm]

        if mode == "losses_Du":
            loss_Du = 0
            with torch.no_grad():
                fake = self.netG(label,edges = edges)
            output_Du_fake = self.netDu(fake)
            loss_Du_fake = self.criterionGAN(output_Du_fake, False).mean()
            loss_Du += loss_Du_fake

            output_Du_real = self.netDu(image)
            loss_Du_real = self.criterionGAN(output_Du_real, True).mean()
            loss_Du += loss_Du_real

            return loss_Du, [loss_Du_fake,loss_Du_real]

        if mode == "generate":
            with torch.no_grad():
                if self.opt.no_EMA:
                    fake = self.netG(label,edges = edges)
                else:
                    fake = self.netEMA(label,edges = edges)
            return fake

        if mode == "segment_real":
            segmentation = self.netD(image)
            return segmentation

        if mode == "segment_fake":
            if self.opt.no_EMA:
                fake = self.netG(label,edges = edges)
            else:
                fake = self.netEMA(label,edges = edges)
            segmentation = self.netD(fake)
            return segmentation

        if mode == "Du_regularize":
            loss_Du = 0
            image.requires_grad = True
            real_pred = self.netDu(image)
            r1_loss = d_r1_loss(real_pred, image).mean()
            loss_Du += 10 * r1_loss
            return loss_Du, [r1_loss]
    
def put_on_multi_gpus(model, opt):
    if opt.gpu_ids != "-1":
        gpus = list(map(int, opt.gpu_ids.split(",")))
        model = DataParallelWithCallback(model, device_ids=gpus).cuda()
    else:
        model.module = model
    assert len(opt.gpu_ids.split(",")) == 0 or opt.batch_size_train % len(opt.gpu_ids.split(",")) == 0
    return model


def preprocess_input(opt, data):
    data['label'] = data['label'].long()
    if opt.gpu_ids != "-1":
        data['label'] = data['label'].cuda()
        data['image'] = data['image'].cuda()
    label_map = data['label']
    bs, _, h, w = label_map.size()
    nc = opt.semantic_nc
    if opt.gpu_ids != "-1":
        input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    else:
        input_label = torch.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return data['image'], input_semantics


def generate_labelmix(label, fake_image, real_image):
    target_map = torch.argmax(label, dim=1, keepdim=True)
    all_classes = torch.unique(target_map)
    for c in all_classes:
        target_map[target_map == c] = torch.randint(0, 2, (1,)).to("cuda")
    target_map = target_map.float()
    mixed_image = target_map * real_image + (1 - target_map) * fake_image
    return mixed_image, target_map
