import argparse

import pandas as pd

import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob

def cal(infer_step,pic_num):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str,
                        default=r'C:\Users\Atlias\Desktop\论文\不同步长推断结果\提取计算 PSNRSSIMFID\\'+str(infer_step))
    args = parser.parse_args()
    real_names = list(glob.glob('{}/*_hr.png'.format(args.path)))
    fake_names = list(glob.glob('{}/*_sr_{}.png'.format(args.path,pic_num)))
    # print(real_names)
    # print(fake_names)
    real_names.sort()
    fake_names.sort()

    for rname, fname in zip(real_names, fake_names):
        hr_img = np.array(Image.open(rname))
        sr_img = np.array(Image.open(fname))
        psnr = Metrics.calculate_psnr(sr_img, hr_img)
        ssim = Metrics.calculate_ssim(sr_img, hr_img)
        print('Image:{}, PSNR:{:.4f}, SSIM:{:.4f}'.format(pic_num, psnr, ssim))
    return psnr,ssim
df_psnr=pd.DataFrame(columns=[str(infer_step) for infer_step in range(200,1200,200)])
df_ssim=pd.DataFrame(columns=[str(infer_step) for infer_step in range(200,1200,200)])
df_psnr.set_index=[pic_num for pic_num in range(1,11)]
df_ssim.set_index=[pic_num for pic_num in range(1,11)]
for infer_step in range(200,1200,200):
    P=[]
    S=[]
    for pic_num in range(1,11):
        psnr,ssim=cal(infer_step,pic_num)
        P.append(psnr)
        S.append(ssim)
    df_psnr[str(infer_step)]=P
    df_ssim[str(infer_step)] = S
df_psnr.to_excel(r'C:\Users\Atlias\Desktop\论文\不同步长推断结果\提取计算 PSNRSSIMFID\psnr.xlsx')
df_ssim.to_excel(r'C:\Users\Atlias\Desktop\论文\不同步长推断结果\提取计算 PSNRSSIMFID\ssim.xlsx')