import itertools
from collections import OrderedDict

import torch
from torch.autograd import Variable

from .util import util as util
from .util.image_pool import ImagePool
from . import networks
from .base_model import BaseModel

from torch import autograd


class CycleGANModelWithRisk(BaseModel):
    def name(self):
        return 'CycleGANModelWithRisk'

    def to_no_grad_var(self, var):
        data = self.as_np(var)
        var = Variable(torch.FloatTensor(data), requires_grad=False)
        if self.gpu_ids:
            var = var.cuda()
        return var


    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.one = torch.FloatTensor([1])
        self.mone = self.one * -1
        if self.opt.gpu_ids:
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()

        self.A_to_B = opt.A_to_B
        self.B_to_A = opt.B_to_A

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)

        self.alligned_input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.alligned_input_B = self.Tensor(nb, opt.output_nc, size, size)

        # load/define networks
        self.netG_A_1 = networks.define_G(opt.input_nc, opt.output_nc,
                                     opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids, num_blocks=opt.num_blocks)
        self.netG_A_2 = networks.define_G(opt.input_nc, opt.output_nc,
                                          opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids, num_blocks=opt.num_blocks)
        self.netG_B_1 = networks.define_G(opt.output_nc, opt.input_nc,
                                    opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids, num_blocks=opt.num_blocks)
        self.netG_B_2 = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, opt.use_dropout, self.gpu_ids, num_blocks=opt.num_blocks)

        use_sigmoid = opt.no_lsgan
        self.netD_A_1 = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        self.netD_A_2 = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        self.netD_B_1 = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        self.netD_B_2 = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)

        if self.opt.correlation_criterion == 'l2':
            self.criterionCorrelation = torch.nn.MSELoss()
        else:
            self.criterionCorrelation = torch.nn.L1Loss()
        self.criterionCorrelation = torch.nn.L1Loss()
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A_1, 'G_A_1', which_epoch)
            self.load_network(self.netG_A_2, 'G_A_2', which_epoch)
            self.load_network(self.netG_B_1, 'G_B_1', which_epoch)
            self.load_network(self.netG_B_2, 'G_B_2', which_epoch)

            self.load_network(self.netD_A_1, 'D_A_1', which_epoch)
            self.load_network(self.netD_A_2, 'D_A_2', which_epoch)
            self.load_network(self.netD_B_1, 'D_B_1', which_epoch)
            self.load_network(self.netD_B_2, 'D_B_2', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr



            self.fake_A_1_pool = ImagePool(opt.pool_size)
            self.fake_A_2_pool = ImagePool(opt.pool_size)
            self.fake_B_1_pool = ImagePool(opt.pool_size)
            self.fake_B_2_pool = ImagePool(opt.pool_size)


            # initialize optimizers
            self.optimizer_G_1 = torch.optim.Adam(itertools.chain(self.netG_A_1.parameters(), self.netG_B_1.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_2 = torch.optim.Adam(itertools.chain(self.netG_A_2.parameters(), self.netG_B_2.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A_1 = torch.optim.Adam(self.netD_A_1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A_2 = torch.optim.Adam(self.netD_A_2.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_B_1 = torch.optim.Adam(self.netD_B_1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B_2 = torch.optim.Adam(self.netD_B_2.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A_1)
            networks.print_network(self.netG_A_2)
            networks.print_network(self.netG_B_1)
            networks.print_network(self.netG_B_2)
            networks.print_network(self.netD_A_1)
            networks.print_network(self.netD_A_2)
            networks.print_network(self.netD_B_1)
            networks.print_network(self.netD_B_2)
            print('-----------------------------------------------')

    def set_input(self, input, alligned_input=None):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # Alligned input here
        alligned_input_A = alligned_input['A' if AtoB else 'B']
        alligned_input_B = alligned_input['B' if AtoB else 'A']
        self.alligned_input_A.resize_(alligned_input_A.size()).copy_(alligned_input_A)
        self.alligned_input_B.resize_(alligned_input_B.size()).copy_(alligned_input_B)
        self.alligned_image_paths = alligned_input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        self.alligned_real_A = Variable(self.alligned_input_A, volatile=True)
        self.alligned_real_B = Variable(self.alligned_input_B, volatile=True)

    def test(self):
        self.alligned_real_A = Variable(self.alligned_input_A, volatile=True)
        self.alligned_fake_B_1 = self.netG_A_1.forward(self.alligned_real_A)
        self.alligned_fake_B_2 = self.netG_A_2.forward(self.alligned_real_A)
        self.alligned_rec_A_1 = self.netG_B_1.forward(self.alligned_fake_B_1)
        self.alligned_rec_A_2 = self.netG_B_2.forward(self.alligned_fake_B_2)

        self.alligned_real_B = Variable(self.input_B, volatile=True)
        self.alligned_fake_A_1 = self.netG_B_1.forward(self.alligned_real_B)
        self.alligned_fake_A_2 = self.netG_B_2.forward(self.alligned_real_B)
        self.alligned_rec_B_1 = self.netG_A_1.forward(self.alligned_fake_A_1)
        self.alligned_rec_B_2 = self.netG_A_2.forward(self.alligned_fake_A_2)

        self.alligned_real_A = Variable(self.alligned_input_A, volatile=True)
        self.alligned_real_B = Variable(self.alligned_input_B, volatile=True)
        # self.alligned_fake_B_1 = self.netG_A_1.forward(self.alligned_real_A)
        self.alligned_fake_B_2 = self.netG_A_2.forward(self.alligned_real_A)
        # self.alligned_fake_A_1 = self.netG_B_1.forward(self.alligned_real_B)
        self.alligned_fake_A_2 = self.netG_B_2.forward(self.alligned_real_B)

        # self.alligned_loss_ground_A_1 = self.criterionCorrelation(self.alligned_fake_A_1, self.alligned_real_A)
        # self.alligned_loss_ground_B_1 = self.criterionCorrelation(self.alligned_fake_B_1, self.alligned_real_B)
        self.alligned_loss_ground_A_2 = self.criterionCorrelation(self.alligned_fake_A_2, self.alligned_real_A)
        self.alligned_loss_ground_B_2 = self.criterionCorrelation(self.alligned_fake_B_2, self.alligned_real_B)

        self.alligned_loss_correlation_A = self.criterionCorrelation(self.alligned_fake_A_1, self.alligned_fake_A_2)
        self.alligned_loss_correlation_B = self.criterionCorrelation(self.alligned_fake_B_1, self.alligned_fake_B_2)

        alligned_pred_fake = self.netD_A_1.forward(self.alligned_fake_B_1)
        self.alligned_loss_G_A_1 = self.criterionGAN(alligned_pred_fake, True)

        alligned_pred_fake = self.netD_A_2.forward(self.alligned_fake_B_2)
        self.alligned_loss_G_A_2 = self.criterionGAN(alligned_pred_fake, True)

        alligned_pred_fake = self.netD_B_1.forward(self.alligned_fake_A_1)
        self.alligned_loss_G_B_1 = self.criterionGAN(alligned_pred_fake, True)

        alligned_pred_fake = self.netD_B_2.forward(self.alligned_fake_A_2)
        self.alligned_loss_G_B_2 = self.criterionGAN(alligned_pred_fake, True)

        # Forward cycle loss
        self.alligned_rec_A_1 = self.netG_B_1.forward(self.alligned_fake_B_1)
        self.alligned_loss_cycle_A_1 = self.criterionCycle(self.alligned_rec_A_1, self.alligned_real_A)

        self.alligned_rec_A_2 = self.netG_B_2.forward(self.alligned_fake_B_2)
        self.alligned_loss_cycle_A_2 = self.criterionCycle(self.alligned_rec_A_2, self.alligned_real_A)

        # Backward cycle loss
        self.alligned_rec_B_1 = self.netG_A_1.forward(self.alligned_fake_A_1)
        self.alligned_loss_cycle_B_1 = self.criterionCycle(self.alligned_rec_B_1, self.alligned_real_B)

        self.alligned_rec_B_2 = self.netG_A_2.forward(self.alligned_fake_A_2)
        self.alligned_loss_cycle_B_2 = self.criterionCycle(self.alligned_rec_B_2, self.alligned_real_B)

        return (self.alligned_loss_ground_A_1, self.alligned_loss_ground_A_2, self.alligned_loss_correlation_A), (self.alligned_loss_ground_B_1, self.alligned_loss_ground_B_2, self.alligned_loss_correlation_B), \
               (self.alligned_loss_G_A_1, self.alligned_loss_G_A_2, self.alligned_loss_G_B_1, self.alligned_loss_G_B_2), (self.alligned_loss_cycle_A_1, self.alligned_loss_cycle_A_2, self.alligned_loss_cycle_B_1, self.alligned_loss_cycle_B_2)

    def backward_D_basic(self, netD, real, fake, dont_train=False):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()

        return loss_D

    def backward_D_A_1(self):
        fake_B_1 = self.fake_B_1_pool.query(self.fake_B_1)
        self.loss_D_A_1 = self.backward_D_basic(self.netD_A_1, self.real_B, fake_B_1)

    def backward_D_A_2(self):
        fake_B_2 = self.fake_B_2_pool.query(self.fake_B_2)
        self.loss_D_A_2 = self.backward_D_basic(self.netD_A_2, self.real_B, fake_B_2)

    def backward_D_B_1(self):
        fake_A_1 = self.fake_A_1_pool.query(self.fake_A_1)
        self.loss_D_B_1 =  self.backward_D_basic(self.netD_B_1, self.real_A, fake_A_1)

    def backward_D_B_2(self):
        fake_A_2 = self.fake_A_2_pool.query(self.fake_A_2)
        self.loss_D_B_2 =  self.backward_D_basic(self.netD_B_2, self.real_A, fake_A_2)

    def backward_G(self):

        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            self.idt_A_1 = self.netG_A_1.forward(self.real_B)
            self.loss_idt_A_1 = self.criterionIdt(self.idt_A_1, self.real_B) * lambda_B * lambda_idt

            self.idt_A_2 = self.netG_A_2.forward(self.real_B)
            self.loss_idt_A_2 = self.criterionIdt(self.idt_A_2, self.real_B) * lambda_B * lambda_idt

            # G_B should be identity if real_A is fed.
            self.idt_B_1 = self.netG_B_1.forward(self.real_A)
            self.loss_idt_B_1 = self.criterionIdt(self.idt_B_1, self.real_A) * lambda_A * lambda_idt

            self.idt_B_2 = self.netG_B_2.forward(self.real_A)
            self.loss_idt_B_2 = self.criterionIdt(self.idt_B_2, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A_1 = 0
            self.loss_idt_B_1 = 0
            self.loss_idt_A_2 = 0
            self.loss_idt_B_2 = 0

        # GAN Loss
        # D_A_1(G_A_1(A))
        self.fake_B_1 = self.netG_A_1.forward(self.real_A)
        pred_fake = self.netD_A_1.forward(self.fake_B_1)
        self.loss_G_A_1 = self.criterionGAN(pred_fake, True)

        # D_A_2(G_A_2(A))
        self.fake_B_2 = self.netG_A_2.forward(self.real_A)
        pred_fake = self.netD_A_2.forward(self.fake_B_2)
        self.loss_G_A_2 = self.criterionGAN(pred_fake, True)

        # D_B_1(G_B_1(B))
        self.fake_A_1 = self.netG_B_1.forward(self.real_B)
        pred_fake = self.netD_B_1.forward(self.fake_A_1)
        self.loss_G_B_1 = self.criterionGAN(pred_fake, True)

        # D_B_2(G_B_2(B))
        self.fake_A_2 = self.netG_B_2.forward(self.real_B)
        pred_fake = self.netD_B_2.forward(self.fake_A_2)
        self.loss_G_B_2 = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        self.rec_A_1 = self.netG_B_1.forward(self.fake_B_1)
        self.loss_cycle_A_1 = self.criterionCycle(self.rec_A_1, self.real_A) * lambda_A

        self.rec_A_2 = self.netG_B_2.forward(self.fake_B_2)
        self.loss_cycle_A_2 = self.criterionCycle(self.rec_A_2, self.real_A) * lambda_A

        # Backward cycle loss
        self.rec_B_1 = self.netG_A_1.forward(self.fake_A_1)
        self.loss_cycle_B_1 = self.criterionCycle(self.rec_B_1, self.real_B) * lambda_B

        self.rec_B_2 = self.netG_A_2.forward(self.fake_A_2)
        self.loss_cycle_B_2 = self.criterionCycle(self.rec_B_2, self.real_B) * lambda_B

        # Correlation loss
        self.loss_correlation_A_1 = - self.opt.lambda_correlation_A * self.criterionCorrelation(self.fake_A_1, self.to_no_grad_var(self.fake_A_2))
        self.loss_correlation_B_1 = - self.opt.lambda_correlation_B * self.criterionCorrelation(self.fake_B_1, self.to_no_grad_var(self.fake_B_2))
        self.loss_correlation_A_2 = - self.opt.lambda_correlation_A * self.criterionCorrelation(self.fake_A_2, self.to_no_grad_var(self.fake_A_1))
        self.loss_correlation_B_2 = - self.opt.lambda_correlation_B * self.criterionCorrelation(self.fake_B_2, self.to_no_grad_var(self.fake_B_1))

        # Combined loss
        self.loss_G_1 = self.loss_G_A_1 + self.loss_G_B_1 + self.loss_cycle_A_1 + self.loss_cycle_B_1 + self.loss_idt_A_1 + self.loss_idt_B_1 #+ self.loss_correlation_A_1 + self.loss_correlation_B_1
        self.loss_G_2 = self.loss_G_A_2 + self.loss_G_B_2 + self.loss_idt_A_2 + self.loss_idt_B_2 + self.loss_correlation_A_2 + self.loss_correlation_B_2 + self.loss_cycle_A_2 + self.loss_cycle_B_2

        self.loss_G_1.backward()
        self.loss_G_2.backward()

        #############################################################################
        ######### Test Values here ##################################################
        #############################################################################

        # Get ground truth losses but do not otpimize accordingly
        self.alligned_fake_B_1 = self.netG_A_1.forward(self.alligned_real_A)
        self.alligned_fake_B_2 = self.netG_A_2.forward(self.alligned_real_A)
        self.alligned_fake_A_1 = self.netG_B_1.forward(self.alligned_real_B)
        self.alligned_fake_A_2 = self.netG_B_2.forward(self.alligned_real_B)

        self.alligned_loss_ground_A_1 = self.criterionCorrelation(self.alligned_fake_A_1, self.alligned_real_A)
        self.alligned_loss_ground_B_1 = self.criterionCorrelation(self.alligned_fake_B_1, self.alligned_real_B)
        self.alligned_loss_ground_A_2 = self.criterionCorrelation(self.alligned_fake_A_2, self.alligned_real_A)
        self.alligned_loss_ground_B_2 = self.criterionCorrelation(self.alligned_fake_B_2, self.alligned_real_B)

        self.alligned_loss_correlation_A = - self.opt.lambda_correlation_A * self.criterionCorrelation(self.alligned_fake_A_1, self.alligned_fake_A_2)
        self.alligned_loss_correlation_B = - self.opt.lambda_correlation_B * self.criterionCorrelation(self.alligned_fake_B_1, self.alligned_fake_B_2)

        # GAN Loss
        # D_A_1(G_A_1(A))
        self.alligned_fake_B_1 = self.netG_A_1.forward(self.alligned_real_A)
        alligned_pred_fake = self.netD_A_1.forward(self.alligned_fake_B_1)
        self.alligned_loss_G_A_1 = self.criterionGAN(alligned_pred_fake, True)

        # D_A_2(G_A_2(A))
        self.alligned_fake_B_2 = self.netG_A_2.forward(self.alligned_real_A)
        alligned_pred_fake = self.netD_A_2.forward(self.alligned_fake_B_2)
        self.alligned_loss_G_A_2 = self.criterionGAN(alligned_pred_fake, True)

        # D_B_1(G_B_1(B))
        self.alligned_fake_A_1 = self.netG_B_1.forward(self.alligned_real_B)
        alligned_pred_fake = self.netD_B_1.forward(self.alligned_fake_A_1)
        self.alligned_loss_G_B_1 = self.criterionGAN(alligned_pred_fake, True)

        # D_B_2(G_B_2(B))
        self.alligned_fake_A_2 = self.netG_B_2.forward(self.alligned_real_B)
        alligned_pred_fake = self.netD_B_2.forward(self.alligned_fake_A_2)
        self.alligned_loss_G_B_2 = self.criterionGAN(alligned_pred_fake, True)

        # Forward cycle loss
        self.alligned_rec_A_1 = self.netG_B_1.forward(self.alligned_fake_B_1)
        self.alligned_loss_cycle_A_1 = self.criterionCycle(self.alligned_rec_A_1, self.alligned_real_A) * lambda_A

        self.alligned_rec_A_2 = self.netG_B_2.forward(self.alligned_fake_B_2)
        self.alligned_loss_cycle_A_2 = self.criterionCycle(self.alligned_rec_A_2, self.alligned_real_A) * lambda_A

        # Backward cycle loss
        self.alligned_rec_B_1 = self.netG_A_1.forward(self.alligned_fake_A_1)
        self.alligned_loss_cycle_B_1 = self.criterionCycle(self.alligned_rec_B_1, self.alligned_real_B) * lambda_B

        self.alligned_rec_B_2 = self.netG_A_2.forward(self.alligned_fake_A_2)
        self.alligned_loss_cycle_B_2 = self.criterionCycle(self.alligned_rec_B_2, self.alligned_real_B) * lambda_B

        # Combined loss
        self.alligned_loss_G_1 = self.alligned_loss_G_A_1 + self.alligned_loss_G_B_1 + self.alligned_loss_cycle_A_1 + self.alligned_loss_cycle_B_1
        self.alligned_loss_G_2 = self.alligned_loss_G_A_2 + self.alligned_loss_G_B_2 + self.alligned_loss_cycle_A_2 + self.alligned_loss_cycle_B_2 + self.alligned_loss_correlation_A + self.alligned_loss_correlation_B

        self.alligned_loss_G_2_A = self.alligned_loss_G_A_2 + self.alligned_loss_cycle_A_2 + self.alligned_loss_correlation_A
        self.alligned_loss_G_2_B = self.alligned_loss_G_B_2 + self.alligned_loss_cycle_B_2 + self.alligned_loss_correlation_B


    def optimize_parameters(self, iter=None):
        # forward
        self.forward()

        # G_A and G_B
        self.optimizer_G_1.zero_grad()
        self.optimizer_G_2.zero_grad()
        self.backward_G()
        self.optimizer_G_1.step()
        self.optimizer_G_2.step()

        # D_A
        if not self.B_to_A:
            self.optimizer_D_A_1.zero_grad()
            self.backward_D_A_1()
            self.optimizer_D_A_1.step()

            self.optimizer_D_A_2.zero_grad()
            self.backward_D_A_2()
            self.optimizer_D_A_2.step()

        # D_B
        if not self.A_to_B:
            self.optimizer_D_B_1.zero_grad()
            self.backward_D_B_1()
            self.optimizer_D_B_1.step()

            self.optimizer_D_B_2.zero_grad()
            self.backward_D_B_2()
            self.optimizer_D_B_2.step()

    def get_current_errors(self, total_steps=None, writer_1=None, writer_2=None):
        D_A_1 = self.loss_D_A_1.item()
        G_A_1 = self.alligned_loss_G_A_1.item()
        Cyc_A_1 = self.alligned_loss_cycle_A_1.item()

        D_B_1 = self.loss_D_B_1.item()
        G_B_1 = self.alligned_loss_G_B_1.item()
        Cyc_B_1 = self.alligned_loss_cycle_B_1.item()

        D_A_2 = self.loss_D_A_2.item()
        G_A_2 = self.alligned_loss_G_A_2.item()
        Cyc_A_2 = self.alligned_loss_cycle_A_2.item()

        D_B_2 = self.loss_D_B_2.item()
        G_B_2 = self.alligned_loss_G_B_2.item()
        Cyc_B_2 = self.alligned_loss_cycle_B_2.item()

        Cycle_loss_A_1 = self.alligned_loss_cycle_A_1.item()
        Cycle_loss_B_1 = self.alligned_loss_cycle_B_1.item()
        Cycle_loss_A_2 = self.alligned_loss_cycle_A_2.item()
        Cycle_loss_B_2 = self.alligned_loss_cycle_B_2.item()

        Corr_loss_A = - self.alligned_loss_correlation_A.item()
        Corr_loss_B = - self.alligned_loss_correlation_B.item()

        Ground_loss_A_1 = self.alligned_loss_ground_A_1.item()
        Ground_loss_B_1 = self.alligned_loss_ground_B_1.item()
        Ground_loss_A_2 = self.alligned_loss_ground_A_2.item()
        Ground_loss_B_2 = self.alligned_loss_ground_B_2.item()

        Test_loss_G_2 = self.alligned_loss_G_2.item()
        Test_loss_G_2_A = self.alligned_loss_G_2_A.item()
        Test_loss_G_2_B = self.alligned_loss_G_2_B.item()

        if self.opt.identity > 0.0:
            idt_A_1 = self.loss_idt_A_1.item()
            idt_B_1 = self.loss_idt_B_1.item()

            idt_A_2 = self.loss_idt_A_2.item()
            idt_B_2 = self.loss_idt_B_2.item()


            return OrderedDict([('D_A_1', D_A_1), ('G_A_1', G_A_1), ('Cyc_A_1', Cyc_A_1), ('idt_A_1', idt_A_1),
                                ('D_B_1', D_B_1), ('G_B_1', G_B_1), ('Cyc_A_2', Cyc_A_2), ('idt_A_2', idt_A_2),
                                ('D_A_2', D_A_2), ('G_A_2', G_A_2), ('Cyc_B_1', Cyc_B_1), ('idt_B_1', idt_B_1),
                                ('D_B_2', D_B_2), ('G_B_2', G_B_2), ('Cyc_B_2', Cyc_B_2), ('idt_B_2', idt_B_2),
                                ("Corr_loss_A", Corr_loss_A), ("Ground_loss_A_1", Ground_loss_A_1), ("Ground_loss_A_2", Ground_loss_A_2),
                                ("Corr_loss_B", Corr_loss_B), ("Ground_loss_B_1", Ground_loss_B_1), ("Ground_loss_B_2", Ground_loss_B_2),
                                ])

        else:

            # # import writer_1
            writer_1.add_scalar("Corr_loss_A", Corr_loss_A, global_step=total_steps)
            writer_1.add_scalar("Corr_loss_B", Corr_loss_B, global_step=total_steps)
            writer_1.add_scalar("Ground_loss_A_1", Ground_loss_A_1, global_step=total_steps)
            writer_1.add_scalar("Ground_loss_A_2", Ground_loss_A_2, global_step=total_steps)
            writer_1.add_scalar("Ground_loss_B_1", Ground_loss_B_1, global_step=total_steps)
            writer_1.add_scalar("Ground_loss_B_2", Ground_loss_B_2, global_step=total_steps)
            writer_1.add_scalar("Cycle_loss_A_1", Cycle_loss_A_1, global_step=total_steps)
            writer_1.add_scalar("Cycle_loss_A_2", Cycle_loss_A_2, global_step=total_steps)
            writer_1.add_scalar("Cycle_loss_B_1", Cycle_loss_B_1, global_step=total_steps)
            writer_1.add_scalar("Cycle_loss_B_2", Cycle_loss_B_2, global_step=total_steps)
            writer_1.add_scalar("G_A_1", G_A_1, global_step=total_steps)
            writer_1.add_scalar("G_A_2", G_A_2, global_step=total_steps)
            writer_1.add_scalar("G_B_1", G_B_1, global_step=total_steps)
            writer_1.add_scalar("G_B_2", G_B_2, global_step=total_steps)
            #
            writer_1.add_scalar("Joint Negative Correlation Loss and Ground Truth Loss 1: A", Corr_loss_A, global_step=total_steps)
            writer_2.add_scalar("Joint Negative Correlation Loss and Ground Truth Loss 1: A", Ground_loss_A_1, global_step=total_steps)
            writer_1.add_scalar("Joint Negative Correlation Loss and Ground Truth Loss 1: B", Corr_loss_B, global_step=total_steps)
            writer_2.add_scalar("Joint Negative Correlation Loss and Ground Truth Loss 1: B", Ground_loss_B_1, global_step=total_steps)


            writer_1.add_scalar("Test Loss and Ground Truth Loss 1:",
                                     (- Test_loss_G_2), global_step=total_steps)

            writer_1.add_scalar("Test Loss and Ground Truth Loss 1 A:",
                                (- Test_loss_G_2_A), global_step=total_steps)

            writer_1.add_scalar("Test Loss and Ground Truth Loss 1 B:",
                                (- Test_loss_G_2_B), global_step=total_steps)

            writer_2.add_scalar("Test Loss and Ground Truth Loss 1:",
                                     (Ground_loss_A_1 + Ground_loss_B_1),
                                     global_step=total_steps)

            writer_2.add_scalar("Test Loss and Ground Truth Loss 1 A:",
                                (Ground_loss_A_1),
                                global_step=total_steps)

            writer_2.add_scalar("Test Loss and Ground Truth Loss 1 B:",
                                (Ground_loss_B_1),
                                global_step=total_steps)

            return OrderedDict([('D_A_1', D_A_1), ('G_A_1', G_A_1), ('Cyc_A_1', Cyc_A_1),
                                ('D_B_1', D_B_1), ('G_B_1', G_B_1), ('Cyc_A_2', Cyc_A_2),
                                ('D_A_2', D_A_2), ('G_A_2', G_A_2), ('Cyc_B_1', Cyc_B_1),
                                ('D_B_2', D_B_2), ('G_B_2', G_B_2), ('Cyc_B_2', Cyc_B_2),
                                ("Corr_loss_A", Corr_loss_A), ("Ground_loss_A_1", Ground_loss_A_1), ("Ground_loss_A_2", Ground_loss_A_2),
                                ("Corr_loss_B", Corr_loss_B), ("Ground_loss_B_1", Ground_loss_B_1), ("Ground_loss_B_2", Ground_loss_B_2),
                                ])

    def get_current_visuals(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.alligned_real_A = Variable(self.alligned_input_A, volatile=True)
        self.alligned_real_B = Variable(self.alligned_input_B, volatile=True)

        real_A = util.tensor2im(self.real_A.data)
        fake_B_1 = util.tensor2im(self.fake_B_1.data)
        fake_B_2 = util.tensor2im(self.fake_B_2.data)
        rec_A_1 = util.tensor2im(self.rec_A_1.data)
        rec_A_2 = util.tensor2im(self.rec_A_2.data)

        real_B = util.tensor2im(self.real_B.data)
        fake_A_1 = util.tensor2im(self.fake_A_1.data)
        fake_A_2 = util.tensor2im(self.fake_A_2.data)
        rec_B_1 = util.tensor2im(self.rec_B_1.data)
        rec_B_2 = util.tensor2im(self.rec_B_2.data)

        alligned_real_A = util.tensor2im(self.alligned_real_A.data)
        alligned_fake_B_1 = util.tensor2im(self.alligned_fake_B_1.data)
        alligned_fake_B_2 = util.tensor2im(self.alligned_fake_B_2.data)

        alligned_real_B = util.tensor2im(self.alligned_real_B.data)
        alligned_fake_A_1 = util.tensor2im(self.alligned_fake_A_1.data)
        alligned_fake_A_2 = util.tensor2im(self.alligned_fake_A_2.data)

        if self.opt.identity > 0.0:
            idt_A_1 = util.tensor2im(self.idt_A_1.data)
            idt_B_1 = util.tensor2im(self.idt_B_1.data)

            idt_A_2 = util.tensor2im(self.idt_A_2.data)
            idt_B_2 = util.tensor2im(self.idt_B_2.data)

            return OrderedDict([#('real_A', real_A), ('fake_B_1', fake_B_1), ('fake_B_2', fake_B_2),
                                #('rec_A_1', rec_A_1), ('rec_A_2', rec_A_2), ('idt_B_1', idt_B_1), ('idt_B_2', idt_B_2),
                                #('real_B', real_B), ('fake_A_1', fake_A_1), ('fake_A_2', fake_A_2),
                                #('rec_B_1', rec_B_1), ('rec_B_2', rec_B_2), ('idt_A_1', idt_A_1), ('idt_A_2', idt_A_2),
                                #('alligned_real_A', alligned_real_A), ('alligned_fake_B_1', alligned_fake_B_1), ('alligned_fake_B_2', alligned_fake_B_2),
                                ('alligned_real_B', alligned_real_B), ('alligned_fake_A_1', alligned_fake_A_1), ('alligned_fake_A_2', alligned_fake_A_2),


                                ])
        else:
            return OrderedDict(
                [('real_A', real_A), ('fake_B_1', fake_B_1), ('fake_B_2', fake_B_2), ('rec_A_1', rec_A_1),
                 ('rec_A_2', rec_A_2),
                 ('real_B', real_B), ('fake_A_1', fake_A_1), ('fake_A_2', fake_A_2), ('rec_B_1', rec_B_1),
                 ('rec_B_2', rec_B_2),
                 ('alligned_real_A', alligned_real_A), ('alligned_fake_B_1', alligned_fake_B_1),
                 ('alligned_fake_B_2', alligned_fake_B_2),
                 ('alligned_real_B', alligned_real_B), ('alligned_fake_A_1', alligned_fake_A_1),
                 ('alligned_fake_A_2', alligned_fake_A_2),

                 ])

            # return OrderedDict(
            #     [
            #      ('alligned_real_A', alligned_real_A), ('alligned_fake_B_1', alligned_fake_B_1),
            #      ('alligned_fake_B_2', alligned_fake_B_2),
            #      ('alligned_real_B', alligned_real_B), ('alligned_fake_A_1', alligned_fake_A_1),
            #      ('alligned_fake_A_2', alligned_fake_A_2),
            #
            #      ])


    def save(self, label):
        self.save_network(self.netG_A_1, 'G_A_1', label, self.gpu_ids)
        self.save_network(self.netD_A_1, 'D_A_1', label, self.gpu_ids)
        self.save_network(self.netG_B_1, 'G_B_1', label, self.gpu_ids)
        self.save_network(self.netD_B_1, 'D_B_1', label, self.gpu_ids)

        self.save_network(self.netG_A_2, 'G_A_2', label, self.gpu_ids)
        self.save_network(self.netD_A_2, 'D_A_2', label, self.gpu_ids)
        self.save_network(self.netG_B_2, 'G_B_2', label, self.gpu_ids)
        self.save_network(self.netD_B_2, 'D_B_2', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A_1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B_1.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_1.param_groups:
            param_group['lr'] = lr

        for param_group in self.optimizer_D_A_2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B_2.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G_2.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_image_paths(self):
        return self.image_paths