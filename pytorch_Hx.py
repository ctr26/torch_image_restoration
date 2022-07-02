

# %%
import torch.nn as nn
import os
import utils
import scipy
from scipy.ndimage import gaussian_filter
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from skimage import color, data, restoration, exposure
from scipy.linalg import circulant
from scipy.sparse import diags
from scipy.signal import convolve2d as conv2
import torch
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import utils
import pyro
from torch.distributions import constraints
import pyro.distributions as dist
import torch.distributions.constraints as constraints
import math
import os
import torch
import torch.distributions.constraints as constraints
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
from scipy.linalg import circulant
from pyro.ops.tensor_utils import convolve
import matplotlib
import pandas as pd
import xarray as xr
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
matplotlib.rcParams['figure.figsize'] = (20, 10)

STEPS = 500
LR = 1e1

results = pd.DataFrame()
results = xr.DataArray()
results = []

# %%
psf_w, psf_h, sigma, scale = 64, 64, 1, 4  # Constants
stop_loss = 1e-2
step_size = 20 * stop_loss / 3.0

astro = (rescale(color.rgb2gray(data.astronaut()), 1.0 / scale)
         * (2**16)/10).astype(int)  # Raw image
psf = np.zeros((psf_w, psf_h))
psf[psf_w // 2, psf_h // 2] = 1
psf = gaussian_filter(psf, sigma=sigma)  # PSF

astro_blur = conv2(astro, psf, "same")  # Blur image
rng = np.random.default_rng()
astro_blur = rng.poisson(astro_blur).astype(int)
y = torch.tensor(astro_blur.flatten().astype('float64')) + 1e-6
# astro_blur = y.reshape(astro_blur.shape)
#  %%
# Function for making circulant matrix because I've not gotten conv2d in torch to work.

H, rolled_psf = utils.make_circulant_from_cropped_psf(
    psf, (psf_w, psf_h), astro.shape
)  # Get H

f = np.asarray(np.matrix(astro_blur.flatten()).transpose()).reshape(
    (-1,)
)  # f is the recorded image
x_0 = np.asarray(np.matmul(H.transpose(), f)).reshape(
    (-1,)
)  # x0 is the initial guess image

# Torch them

H_torch = torch.tensor(
    H, dtype=torch.float
).to_sparse()  # This is sparse, utterlly full of zeros, speeds everything up
b_torch = torch.tensor(f, dtype=torch.float)
x_torch = torch.tensor(
    x_0, requires_grad=True, dtype=torch.float
)  # Requires grad is magic

# print('Loss before: %s' % (torch.norm(torch.matmul(H_torch, x_torch) - b_torch)))
def p_x_given_b(b, Ax):
    # This is log(p(x|b))
    b = b + 1e-6
    Ax = Ax + 1e-6
    log_b_factorial = torch.lgamma(b+1)
    return torch.multiply(torch.log(Ax), (b)) - Ax - log_b_factorial


def log_liklihood_x_given_b(b, Ax):
    return -torch.sum(
        p_x_given_b(b, Ax)
    )

def aesthics(astro,comparsion,method,iteration):
    return pd.DataFrame(
            {
            "method":method,
            "mse_gt":mean_squared_error(comparsion, astro),
            "psnr_gt":peak_signal_noise_ratio(comparsion, astro,data_range=astro.max() - astro.min()),
            # "ssim_gt":ssim(np.array(comparsion).astype(np.float64),
            #             np.array(astro)).astype(np.float64),
            "ssim_gt":ssim(comparsion,astro,data_range=astro.max() - astro.min()),
            "iteration":iteration
            },index=[0])
#  %%
class HTorch(nn.Module):
    def __init__(self, H):
        super(HTorch, self).__init__()
        self.H = H
        # self.x_0 = x_0
        # self.x = torch.nn.Linear(H.shape[0], 1)
        self.x_torch = torch.tensor(
                x_0, requires_grad=True, dtype=torch.float
            )  # Requires grad is magic
        self.x = torch.nn.Parameter(self.x_torch)
    # def __call__(self,x):
    #     return torch.matmul(H_torch, x).double()
    def forward(self):
        # self.x = torch.nn.Parameter((self.x>0)*self.x)
        return torch.matmul(H_torch, self.x).double()
    def predict(self,x):
        # self.x = torch.nn.Parameter((self.x>0)*self.x)
        return torch.matmul(H_torch.double(), x.double()).double()



model = HTorch(H_torch)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
mse_pred_list = []
pbar = tqdm(range(STEPS))
pbar = tqdm(range(50))

for t in pbar:
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(1, torch.inf)
    y_pred = model()
    loss = loss_fn(y_pred, y.double())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"loss": f'{loss:.2f}'})
    # scheduler.step(loss)
    # print(optimizer.state_dict()["param_groups"][0]["lr"])
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(0, torch.inf)
    mse_guess = model.x.detach().numpy().reshape(astro.shape)
    results.append(aesthics(astro,mse_guess,"mse",t))

    # results["mse_guess"] = xr.DataArray(
    #     mse_guess, coords={"Iteration": t, "Kind": "mse_guess"})
plt.imshow(mse_guess)
plt.show()
# %%


x_torch = torch.tensor(
    x_0, requires_grad=True, dtype=torch.float
)  # Requires grad is magic


class HTorch_VT(nn.Module):
    def __init__(self, H):
        super(HTorch_VT, self).__init__()
        self.H = H
        # self.x_0 = x_0
        # self.x = torch.nn.Linear(H.shape[0], 1)
        self.x_torch = torch.tensor(
                x_0, requires_grad=True, dtype=torch.float
            )  # Requires grad is magic
        self.x = torch.nn.Parameter(self.x_torch)

        self.img_T = torch.tensor(np.random.binomial(y.double(),0.5)).double().requires_grad_()
        self.img_V = y.double() - self.img_T
    # def __call__(self,x):
    #     return torch.matmul(H_torch, x).double()
    def forward(self):
        # self.x = (self.x>0)*self.x


        # self.img_T = torch.tensor(np.random.binomial(y.double(),0.5)).double().requires_grad_()
        # # self.img_T = torch.tensor(np.random.binomial(y.double(),0.5)).double()

        self.img_T = torch.tensor(np.random.binomial(y.double(),0.5)).double().requires_grad_()


        # img_V = y.double() - self.img_T
        y_pred_T = self.predict(self.img_T.double())+1e-6-y.double()
        y_pred_V = self.predict(self.img_V.double())+1e-6-y.double()

        return [self.predict(self.x).double(),y_pred_T,y_pred_V]
        # return (torch.matmul(H_torch, self.x).float(),np.array([0]),np.array([0]))
        # return torch.matmul(H_torch, self.x).double()

    def predict(self,x):
        # self.x = torch.nn.Parameter((self.x>0)*self.x)
        return torch.matmul(H_torch.float(), x.float()).float()
    # def get_static_V_T(self):


model = HTorch_VT(H_torch)
loss_fn = torch.nn.MSELoss()
loss_fn_poisson = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
mse_pred_list = []
pbar = tqdm(range(STEPS))
pbar = tqdm(range(500))
pbar = tqdm(range(100))

for lamba_reg in [1,10,100,1000]:
    for t in pbar:
        # y_pred = model()
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(1, torch.inf)
        y_pred,y_pred_T,y_pred_V = model()

        # img_T = torch.tensor(np.random.binomial(y.double(),0.5))
        # img_V = y.double() - img_T

        # y_pred_T = model.predict(img_T.double())+1e-6
        # y_pred_V = model.predict(img_V.double())+1e-6

        # poisson_thin_guess_loss = torch.nn.functional.cosine_similarity(
        #     torch.sqrt((y_pred_T-y_pred)**2).ravel(),
        #     torch.sqrt((y_pred_V-y_pred)**2).ravel(),
        #     dim=0);poisson_thin_guess_loss

        # poisson_thin_guess_loss = (torch.nn.functional.cosine_similarity(
        #     (y_pred_T-y_pred).ravel().unsqueeze(0),
        #     (y_pred_V-y_pred).ravel().unsqueeze(0),
        #     dim=0)>0).int();poisson_thin_guess_loss

        # poisson_thin_guess_loss = -(torch.nn.functional.cosine_similarity(
        #     (y_pred_T).ravel().unsqueeze(0),
        #     (y_pred_V).ravel().unsqueeze(0),
        #     dim=0)>0).int();poisson_thin_guess_loss

        # poisson_thin_guess_loss = (torch.nn.functional.cosine_similarity(
        #     (y_pred_T).ravel().unsqueeze(0),
        #     (y_pred_V).ravel().unsqueeze(0),
        #     dim=0)>0);poisson_thin_guess_loss

        # poisson_thin_guess_loss = (y_pred_T*y_pred_V)>0
        # poisson_thin_guess_loss = torch.nn.MSELoss()(
        #     (p_x_given_b(y_pred_T,y_pred)).ravel().unsqueeze(0),
        #     (p_x_given_b(y_pred_V,y_pred)).ravel().unsqueeze(0),
        #     );poisson_thin_guess_loss
            

        poisson_thin_guess_loss = torch.nn.MSELoss()(
            y_pred_T,y_pred,
            );poisson_thin_guess_loss

        poisson_thin_guess_loss = torch.nn.MSELoss()(
            y_pred_T-y_pred_V,y_pred,
            );poisson_thin_guess_loss
        # poisson_thin_guess_loss_T = torch.nn.MSELoss()(
        #     y_pred_T,y_pred,
        #     );poisson_thin_guess_loss
        # poisson_thin_guess_loss_V = torch.nn.MSELoss()(
        #     y_pred_V,y_pred,
        #     );poisson_thin_guess_loss
        # poisson_thin_guess_loss = poisson_thin_guess_loss_T+poisson_thin_guess_loss_V
        # poisson_thin_guess_loss = 0
        # poisson_thin_guess_loss = torch.nn.MSELoss()(
        #     torch.sqrt((y_pred_T-y_pred)**2).ravel(),
        #     torch.sqrt((y_pred_V-y_pred)**2).ravel());poisson_thin_guess_loss


        # loss = loss_fn(y_pred*poisson_thin_guess_loss,y*poisson_thin_guess_loss).double()
        
        # loss = loss_fn(y_pred*poisson_thin_guess_loss,
        #             y.double()*poisson_thin_guess_loss)
        loss = loss_fn(y_pred,y.double())+poisson_thin_guess_loss/lamba_reg
        # loss = poisson_thin_guess_loss

        # loss =  poisson_thin_guess_loss

        # loss = log_liklihood_x_given_b(b_torch, y_pred)*log_liklihood_x_given_b(y_pred_T, y_pred_V)
        # loss = log_liklihood_x_given_b(b_torch, y_pred)+poisson_thin_guess_loss

        # loss = torch.nn.MSELoss()(y_pred, y.double())
        # y_pred
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(
                {"loss": f'{loss:.2f}',
                "poisson_thin_guess_loss": f'{poisson_thin_guess_loss:.2f}'}
                )
        # pbar.set_postfix(
        #         {"loss": f'{loss:.2f}'}
        #         )
        # scheduler.step(loss)
        # print(optimizer.state_dict()["param_groups"][0]["lr"])
        with torch.no_grad():
            for param in model.parameters():
                param.clamp_(0, torch.inf)
        poisson_thin_guess = model.x.detach().numpy().reshape(astro.shape)
        # results["mse_guess"] = xr.DataArray(
        #     mse_guess, coords={"Iteration": t, "Kind": "mse_guess"})
        results.append(aesthics(astro,poisson_thin_guess,f"poisson_thin-{lamba_reg}",t))
    plt.imshow(poisson_thin_guess)
    plt.show()
# %%

H_torch = torch.tensor(
    H, dtype=torch.float
).to_sparse()  # This is sparse, utterlly full of zeros, speeds everything up
b_torch = torch.tensor(f, dtype=torch.float)
x_torch = torch.tensor(
    x_0, requires_grad=True, dtype=torch.float
)  # Requires grad is magic

# For Poisson noise


y_pred = torch.matmul(H_torch, b_torch)

# lr = torch.sqrt(torch.norm(p_x_given_b(b_torch+1e-6, y_pred))); lr
# lr = 1000
lr = LR

model = HTorch(H_torch)
loss_fn = log_liklihood_x_given_b
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

mle_pred_list = []
# %%
pbar = tqdm(range(STEPS))
pbar = tqdm(range(50))

for t in pbar:
    y_pred = model()

    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(0.0001, torch.inf)
    # loss = loss_fn(y.double(),y_pred)
    x_old = model.x.detach().numpy()
    H_T_1 = torch.matmul(torch.t(H_torch),torch.ones_like(x_torch))
    lr = torch.norm(x_old/H_T_1)/len(x_old)
    # lr = torch.mean(x_old/H_T_1)
    # lr = torch.norm((x_old/H_T_1)/len(x_old))

    # Factor of 2 for nyquist sampling of the loss step
    lr = lr/2; lr
    # lr = torch.norm(p_x_given_b(b_torch,torch.matmul(H_torch, b_torch)))
    loss = loss_fn(b_torch, y_pred)
    loss
    # loss = loss_fn(y_pred,b_torch)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for param in model.parameters():
            param.clamp_(0.0001, torch.inf)
    pbar.set_postfix({"loss": f'{loss:.2f}',
                      "y_pred": f'{y_pred.min().detach().numpy():.2f}',
                      "lr": f'{lr:.2f}',
                      })
    mle_guess = model.x.detach().numpy().reshape(astro.shape)
    results.append(aesthics(astro,mle_guess,"mle",t))

    # results["mle_guess"] = mle_guess


# %%

# y = (H*x)+n
# @pyro.condition(data={"Hx": y})


def model(y, H):
    x = pyro.param("x", torch.tensor(x_0).float(),
                   constraint=constraints.positive)
    Hx = torch.matmul(H, x).float()
    pyro.sample("f", dist.Poisson(Hx).to_event(1), obs=y)

# MLE means we can ignore the guide function
# MAP would mean that the guide (prior/posteri?) would be a delta function


def guide(y, H):
    pass

# def guide(y,H):
#     f_map = pyro.param("f_map", torch.tensor(0.5),
#                        constraint=constraints.unit_interval)
#     pyro.sample("latent_fairness", dist.Delta(f_map))


# Setup the optimizer
adam_params = {"lr": 0.01}
optimizer = Adam(adam_params)

# Setup the inference algorithm
# https://en.wikipedia.org/wiki/Evidence_lower_bound
svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

loss = []
pyro.clear_param_store()
n_steps = 100

mle_pred_list = []

pbar = tqdm(range(STEPS))
pbar = tqdm(range(50))

for step in pbar:
    loss.append(svi.step(y.int(), torch.tensor(H).float()))
    # y_pred = pyro.param("x").detach().numpy()
    pbar.set_postfix({"loss": f'{loss[-1]:.2f}'})
    # print(loss[-1])
    svi_guess = pyro.param("x").detach().numpy().reshape(astro.shape)
    # results["svi_guess"] = svi_guess
    results.append(aesthics(astro,svi_guess,"SVI",step))

# svi_guess = pyro.param("f").detach().numpy().reshape(astro.shape)
# %%

import seaborn as sns
df = pd.concat(results)

sns.lmplot(x="iteration",y="ssim_gt",hue="method",data=df,sharey=True,fit_reg=False)
plt.show()

sns.lmplot(x="iteration",y="mse_gt",hue="method",data=df,sharey=True,fit_reg=False)
plt.show()

sns.lmplot(x="iteration",y="psnr_gt",hue="method",data=df,sharey=True,fit_reg=False)
plt.show()


# %%

fig, ax = plt.subplots(2, 4)
ax[0, 0].imshow(astro).axes.set_title("astro")
ax[0, 1].imshow(astro_blur).axes.set_title("astro_blur")
ax[0, 2].set_axis_off()

ax[1, 0].imshow(mse_guess).axes.set_title("mse_guess")
ax[1, 2].imshow(poisson_thin_guess).axes.set_title("poisson_thin_guess")
ax[1, 1].imshow(mle_guess).axes.set_title("mle_guess")
ax[1, 3].imshow(svi_guess).axes.set_title("svi_guess")
plt.tight_layout()
plt.show()

# %%



{"mse_mse":mean_squared_error(mse_guess, astro),
"mse_mle":mean_squared_error(mle_guess, astro),
"mse_svi":mean_squared_error(svi_guess, astro),
"poisson_thin_guess":mean_squared_error(poisson_thin_guess, astro),
}
# %%
{"mse_mse":ssim(mse_guess, astro),
"mse_mle":ssim(mle_guess, astro),
"mse_svi":ssim(svi_guess, astro),
"poisson_thin_guess":ssim(poisson_thin_guess, astro),
}

# %%

H_torch = torch.tensor(
    H, dtype=torch.float
).to_sparse()  # This is sparse, utterlly full of zeros, speeds everything up
b_torch = torch.tensor(f, dtype=torch.float)
x_torch = torch.tensor(
    x_0, requires_grad=True, dtype=torch.float
)  # Requires grad is magic

# For Poisson noise


y_pred = torch.matmul(H_torch, b_torch)

# Learning rate is length of the vector
lr = torch.sqrt(torch.linalg.norm(p_x_given_b(b_torch, y_pred)))
# lr = 1000
lr

model = HTorch(H_torch)
loss_fn = log_liklihood_x_given_b
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

mle_pred_list = []

pbar = tqdm(range(40))
for t in pbar:

    y_pred = model()

    # lr = torch.norm(p_x_given_b(b_torch,torch.matmul(H_torch, b_torch)))
    loss = loss_fn(b_torch, y_pred)
    
    # with torch.no_grad():
    #     for param in model.parameters():
    #         param.clamp_(0.0001, torch.inf)
    #             # loss = loss_fn(y.double(),y_pred)
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            x_old = param.data
            param.clamp_(1, torch.inf)
            H_T_1 = torch.matmul(torch.t(H_torch),torch.ones_like(x_torch))
            lr = torch.norm(x_old/H_T_1)/len(x_old)
            param.data -= lr * param.grad.data
            param.clamp_(1, torch.inf)


    # optimizer.param_groups[0]['lr'] = lr
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # with torch.no_grad():
        # for param in model.parameters():
            # param.clamp_(0.0001, torch.inf)
    pbar.set_postfix({"loss": f'{loss:.2f}',
                      "y_pred": f'{y_pred.min().detach().numpy():.2f}',
                      "lr": f'{lr:.2f}',
                      })
    mle_guess = model.x.detach().numpy().reshape(astro.shape)
    # results["mle_guess"] = mle_guess
plt.imshow(model.x.detach().numpy().reshape(astro.shape))

# %%
