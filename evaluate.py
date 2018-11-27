import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import sys

res_file = sys.argv[1]
gt_file = sys.argv[2]
ground_truth = misc.imread(gt_file).astype(float)
result = misc.imread(res_file).astype(float)
print(np.shape(ground_truth))
print(np.shape(result))
c1 = (0.01*255)**2
c2 = (0.03*255)**2
W = 32 # window size
H = 32 # Hop size
M = np.shape(result)[0]
N = np.shape(result)[1]
# Divide result image into windows
res_blocks = np.zeros((M/H,N/H,W,W,3))
for u in range(0,M-H,H):
	for v in range(0,N-H,H):
		res_blocks[u/H,v/H] = result[u:u+W,v:v+W]
# Compute mean and variance of window
mean1 = np.mean(res_blocks,(2,3,4))
var1 = np.var(res_blocks,(2,3,4))
ssim_map = np.zeros((M/H,N/H))

#  Calculate MSE and PSNR
MSE = np.mean((ground_truth-result)**2,(0,1,2))
psnr = 10*np.log10(255*255/MSE)
# Divide ground truth image into windows
gt_blocks = np.zeros((M/H,N/H,W,W,3))
for u in range(0,M-H,H):
	for v in range(0,N-H,H):
		gt_blocks[u/H,v/H] = ground_truth[u:u+W,v:v+W].copy()
# Compute mean and variance of ground truth image and covariance
mean2 = np.mean(gt_blocks,(2,3,4))
var2 = np.var(gt_blocks,(2,3,4))
covar = np.mean( (res_blocks-np.reshape(mean1,[M/H,N/H,1,1,1]))
	*(gt_blocks-np.reshape(mean2,[M/H,N/H,1,1,1])) , (2,3,4))
# Find SSIM for each window
ssim_map = (2*mean1*mean2+c1)*(2*covar+c2)/(mean1**2+mean2**2+c1)/(var1+var2+c2)
# Find average SSIM
ssim = np.mean(ssim_map,(0,1))
# Report SSIM and PSNR
print("PSNR = "+str(psnr))
print("SSIM = "+str(ssim))