import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"])
            self.log_dict = OrderedDict()
        self.load_network()
        self.print_network()
        # from thop import profile
        # input = torch.randn(1, 3, 256, 256)
        # macs, params = profile(self.netG, inputs=(input,))
        # print('!!'*60,macs,params)

    def feed_data(self, data):
        self.data = self.set_device(data)

    # def _warmup_beta(self,linear_start, linear_end, n_timestep, warmup_frac):
    #     betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    #     warmup_time = int(n_timestep * warmup_frac)
    #     betas[:warmup_time] = np.linspace(
    #         linear_start, linear_end, warmup_time, dtype=np.float64)
    #     return betas
    # def make_beta_schedule(self,schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    #     if schedule == 'quad':
    #         betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
    #                             n_timestep, dtype=np.float64) ** 2
    #     elif schedule == 'linear':
    #         betas = np.linspace(linear_start, linear_end,
    #                             n_timestep, dtype=np.float64)
    #     elif schedule == 'warmup10':
    #         betas = self._warmup_beta(linear_start, linear_end,
    #                              n_timestep, 0.1)
    #     elif schedule == 'warmup50':
    #         betas = self._warmup_beta(linear_start, linear_end,
    #                              n_timestep, 0.5)
    #     elif schedule == 'const':
    #         betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    #     elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    #         betas = 1. / np.linspace(n_timestep,
    #                                  1, n_timestep, dtype=np.float64)
    #     elif schedule == "cosine":
    #         timesteps = (
    #                 torch.arange(n_timestep + 1, dtype=torch.float64) /
    #                 n_timestep + cosine_s
    #         )
    #         alphas = timesteps / (1 + cosine_s) * math.pi / 2
    #         alphas = torch.cos(alphas).pow(2)
    #         alphas = alphas / alphas[0]
    #         betas = 1 - alphas[1:] / alphas[:-1]
    #         betas = betas.clamp(max=0.999)
    #     else:
    #         raise NotImplementedError(schedule)
    #     return betas

    def optimize_parameters(self):

        # betas = self.make_beta_schedule(
        #     schedule="linear",
        #     n_timestep=1000,
        #     linear_start=1e-6,
        #     linear_end=1e-2)
        # betas = betas.detach().cpu().numpy() if isinstance(
        #     betas, torch.Tensor) else betas
        # alphas = 1. - betas
        # alphas_cumprod = np.cumprod(alphas, axis=0)
        # alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        # self.sqrt_alphas_cumprod_prev = np.sqrt(
        #     np.append(1., alphas_cumprod))
        #
        # x_start = self.data['HR']
        # [b, c, h, w] = x_start.shape
        # t = np.random.randint(1, 1000 + 1)
        # continuous_sqrt_alpha_cumprod = torch.FloatTensor(
        #     np.random.uniform(
        #         self.sqrt_alphas_cumprod_prev[t-1],
        #         self.sqrt_alphas_cumprod_prev[t],
        #         size=b
        #     )
        # ).to(x_start.device)
        # continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
        #     b, -1)
        # from thop import profile
        # macs, params = profile(self.netG, inputs=(self.data,continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1)))
        # print('!!' * 60, macs, params)
        # print('FLOPs = ' + str(macs / 1000 ** 3) + 'G')
        # print('Params = ' + str(params / 1000 ** 2) + 'M')

        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        l_pix = l_pix.sum()/int(b*c*h*w)
        l_pix.backward()
        self.optG.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()

    def test(self, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['SR'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['SR'], continous)
        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['SR'] = self.SR.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))
            # network.load_state_dict(torch.load(
            #     gen_path), strict=False)
            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
