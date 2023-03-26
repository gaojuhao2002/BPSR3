#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: GJH
# @file: load_and_eval
# @time: 2023/3/21,21:59
import pandas as pd
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)
def load_and_infer(ckpt_path,val_len):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/infer_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_false')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    opt['path']['checkpoint']=ckpt_path#变成ckpt的路径
    print('*-'*60,opt['path']['checkpoint'])
    opt['datasets']['val']['data_len']=val_len

    # print('-*'*20,'测试集数据长度：',opt['datasets']['val']['data_len'])
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Logger.setup_logger(None, opt['path']['log'],
    #                     'train', level=logging.INFO, screen=True)
    # Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    # logger.info(Logger.dict2str(opt))

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Val Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    # current_step = opt['iter']
    # current_epoch= opt['epoch']
    idx = 0
    avg_psnr = 0.0
    avg_ssim = 0.0
    print('测试集数量:',len(val_loader))
    # os.makedirs(result_path, exist_ok=True)
    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8

        logger.info('Begin Model Evaluation.')

        eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
        eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
        Metrics.save_img(
            Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format('temp', 0, idx))
        avg_psnr += eval_psnr
        avg_ssim += eval_ssim

    avg_psnr = avg_psnr / idx
    avg_ssim = avg_ssim / idx
    # logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
    # logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
    # logger_val = logging.getLogger('val')  # validation logger
    # logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
    #     current_epoch, current_step, avg_psnr, avg_ssim))
    return avg_psnr,avg_ssim
def get_file_names(path):
    file_names = os.listdir(path)
    processed_names = []
    for name in file_names:
        suffix = os.path.splitext(name)[1]
        if suffix == '.pth':
            index = name.find('_gen.pth')
            if index == -1:
                index = name.find('_opt.pth')
            processed_name = name[:index]
            if processed_name not in processed_names:
                processed_names.append(processed_name)
    return processed_names
if __name__=='__main__':
    ckpt_path='/SR3/4x_diff_multry/Image-Super-Resolution-via-Iterative-Refinement-master/experiments/64_256_Train_0iter_230318_031257/checkpoint/'
    ckpt_list=get_file_names(ckpt_path)

    t_dict={}
    for k,epoch_name in enumerate(ckpt_list):
        index_of_e = epoch_name.index('E')
        epoch = epoch_name[index_of_e + 1:]
        t_dict[(int(epoch))]=epoch_name
    ckpt_list = {k: t_dict[k] for k in sorted(t_dict.keys())}.values()

    df=pd.DataFrame(columns=['PSNR','SSIM'])
    df_temp=pd.DataFrame(columns=['PSNR','SSIM'])
    for k,epoch_name in enumerate(ckpt_list):
        index_of_e = epoch_name.index('E')
        epoch = epoch_name[index_of_e + 1:]
        print(epoch_name)
        avg_psnr,avg_ssim=load_and_infer(ckpt_path+epoch_name,val_len=3)
        df.loc[epoch]=[avg_psnr,avg_ssim]
        df_temp.loc[epoch]=[avg_psnr,avg_ssim]
        if k%1==0:
            df_temp.to_excel('/SR3/Only_Infer_result/DBPN_RESULT/res_{}.xlsx'.format(k))
    df.to_excel('/SR3/Only_Infer_result/DBPN_RESULT/ffhq_100_0witer_res.xlsx')



    ckpt_path='/SR3/4x_diff_multry/Image-Super-Resolution-via-Iterative-Refinement-master/experiments/64_256_Train_300000iter_230320_080415/checkpoint/'
    ckpt_list=get_file_names(ckpt_path)
    t_dict={}
    for k,epoch_name in enumerate(ckpt_list):
        index_of_e = epoch_name.index('E')
        epoch = epoch_name[index_of_e + 1:]
        t_dict[(int(epoch))]=epoch_name
    ckpt_list = {k: t_dict[k] for k in sorted(t_dict.keys())}.values()

    df=pd.DataFrame(columns=['PSNR','SSIM'])
    df_temp=pd.DataFrame(columns=['PSNR','SSIM'])
    for k,epoch_name in enumerate(ckpt_list):
        index_of_e = epoch_name.index('E')
        epoch = epoch_name[index_of_e + 1:]
        print(epoch_name)
        avg_psnr,avg_ssim=load_and_infer(ckpt_path+epoch_name,val_len=10)
        df.loc[epoch]=[avg_psnr,avg_ssim]
        df_temp.loc[epoch]=[avg_psnr,avg_ssim]
        if k%1==0:
            df_temp.to_excel('/SR3/Only_Infer_result/DBPN_RESULT_30w/res_{}.xlsx'.format(k))
    df.to_excel('/SR3/Only_Infer_result/DBPN_RESULT_30w/ffhq_100_0witer_res.xlsx')