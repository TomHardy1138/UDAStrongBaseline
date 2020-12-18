import torch
import torch.nn as nn
import time
import numpy as np

from UDAsbs.multigpu import gpu_mul_Ax, gpu_mul_xA, aggreg_multi_gpu, gpu_mul_AB

from scipy.special import logsumexp

def py_softmax(x, axis=None):
    """stable softmax"""
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

def cpu_sk(self):
    """ Sinkhorn Knopp optimization on CPU
        * stores activations to RAM
        * does matrix-vector multiplies on CPU
        * slower than GPU
    """
    # 1. aggregate inputs:
    self.model.eval()


    N = len(self.pseudo_loader.dataset)
    if self.hc == 1:
        self.PS = np.zeros((N, self.K[0]), dtype=self.dtype)
    else:
        self.PS_pre = np.zeros((N, self.presize), dtype=self.dtype)
    now = time.time()
    for batch_idx, item in enumerate(self.pseudo_loader):
        data = item[0].to(self.dev)
        if self.hc == 1:
            _,predicted,_=self.model(data,training=True)# predicted=self.model(data,cluster=True)
            p = nn.functional.softmax(predicted[0], 1)
            self.PS[item[-1], :] = p.detach().cpu().numpy().astype(self.dtype)
        else:
            self.model.headcount = self.hc
            p = self.model(data)
            self.PS_pre[item[-1], :] = p.detach().cpu().numpy().astype(self.dtype)
    print("Aggreg of outputs  took {0:.2f} min".format((time.time() - now) / 60.), flush=True)


    # 2. solve label assignment via sinkhorn-knopp:
    if self.hc == 1:
        optimize_L_sk(self, nh=0)
    else:
        for nh in range(self.hc):
            print("computing head %s " % nh, end="\r", flush=True)
            tl = getattr(self.model.module, "classifier{}_{}".format(nh,self.K[nh]))
            time_mat = time.time()
            # clear memory
            try:
                del self.PS
            except:
                pass

            # apply last FC layer (a matmul and adding of bias)
            self.PS = self.PS_pre @ tl.weight.cpu().detach().numpy().T.astype(self.dtype)
            # + tl.bias.cpu().detach().numpy().astype(self.dtype))
            print("matmul took %smin" % ((time.time() - time_mat) / 60.), flush=True)
            self.PS = py_softmax(self.PS, 1)
            optimize_L_sk(self, nh=nh)
    return

def gpu_sk(self):
    """ Sinkhorn Knopp optimization on GPU
            * stores activations on multiple GPUs (needed when dataset is large)
            * does matrix-vector multiplies on GPU (extremely fast)
            * recommended variant
            * due to multi-GPU use, it's a bit harder to understand what's happening -> see CPU variant to understand
    """
    # 1. aggregate inputs:
    start_t = time.time()
    if self.hc == 1:
        self.PS, indices = aggreg_multi_gpu(self.model, self.pseudo_loader,
                                            hc=self.hc, dim=self.outs[0], TYPE=self.dtype)
    else:
        try: # just in case stuff
            del self.PS_pre
        except:
            pass
        torch.cuda.empty_cache()
        time.sleep(1)
        self.PS_pre, indices = aggreg_multi_gpu(self.model, self.pseudo_loader,
                                                hc=self.hc, dim=self.presize, TYPE=torch.float32)

        self.model.headcount = self.hc
    print("Aggreg of outputs  took {0:.2f} min".format((time.time() - start_t) / 60.), flush=True)

    # 2. solve label assignment via sinkhorn-knopp:
    if self.hc == 1:
        optimize_L_sk_multi(self, nh=0)
        self.L[0,indices] = self.L[0,:]
    else:
        for nh in range(self.hc):
            tl = getattr(self.model, "top_layer%d" % nh)
            time_mat = time.time()
            try:
                del self.PS
                torch.cuda.empty_cache()
            except:
                pass

            # apply last FC layer (a matmul and adding of bias)
            self.PS = gpu_mul_AB(self.PS_pre, tl.weight.t(),
                                 c=tl.bias, dim=self.outs[nh], TYPE=self.dtype)
            print("matmul took %smin" % ((time.time() - time_mat) / 60.), flush=True)
            optimize_L_sk_multi(self, nh=nh)
            self.L[nh][indices] = self.L[nh]
    return
import collections
def optimize_L_sk(self, nh=0):
    N = max(self.L[nh].size())
    tt = time.time()
    self.PS = self.PS.T # now it is K x N


    if not self.dis_gt:
        r = np.ones((self.outs[nh], 1), dtype=self.dtype) / self.outs[nh]
    else:
        b_pesud_label = np.nanargmax(self.PS, 0)
        plabel2number=dict(collections.Counter(b_pesud_label)).items()
        plabel2number=sorted(plabel2number,key=lambda plabel2number:plabel2number[1])
        sort_label=[label[0] for label in plabel2number]
        origin_dis=self.dis_gt
        deta=len(origin_dis)/ self.outs[nh]
        r = np.ones((self.outs[nh], 1), dtype=self.dtype) / N
        for i,sl in enumerate(sort_label[::-1]):
            nn=origin_dis[0 + int(round(i * deta))]
            r[sl,:] = nn
        r=py_softmax(r,axis=0)

    c = np.ones((N, 1), dtype=self.dtype) / N
    self.PS **= self.lamb  # K x N
    inv_K = self.dtype(1./self.outs[nh])
    inv_N = self.dtype(1./N)
    err = 1e6
    _counter = 0

    while err > 1e-2:
        r = inv_K / (self.PS @ c)          # (KxN)@(N,1) = K x 1
        c_new = inv_N / (r.T @ self.PS).T  # ((1,K)@(KxN)).t() = N x 1
        if _counter % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)  # " nonneg: ", sum(I), flush=True)
    # inplace calculations.
    self.PS *= np.squeeze(c)
    self.PS = self.PS.T
    self.PS *= np.squeeze(r)
    self.PS = self.PS.T
    argmaxes = np.nanargmax(self.PS, 0) # size N
    newL = torch.LongTensor(argmaxes)
    self.L[nh] = newL.to(self.dev)
    print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)



def optimize_L_sk_multi(self, nh=0):
    """ optimizes label assignment via Sinkhorn-Knopp.

         this implementation uses multiple GPUs to store the activations which allow fast matrix multiplies

         Parameters:
             nh (int) number of the head that is being optimized.

    """
    N = max(self.L.size())
    tt = time.time()
    r = torch.ones((self.outs[nh], 1), device='cuda:0', dtype=self.dtype) / self.outs[nh]
    c = torch.ones((N, 1), device='cuda:0', dtype=self.dtype) / N
    ones = torch.ones(N, device='cuda:0', dtype=self.dtype)
    inv_K = 1. / self.outs[nh]
    inv_N = 1. / N

    # inplace power of softmax activations:
    [qq.pow_(self.lamb) for qq in self.PS]  # K x N

    err = 1e6
    _counter = 0
    ngpu = torch.cuda.device_count()
    splits = np.cumsum([0] + [a.size(0) for a in self.PS])
    while err > 1e-1:
        r = inv_K / (gpu_mul_xA(c.t(), self.PS,
                                ngpu=ngpu, splits=splits, TYPE=self.dtype)).t()  # ((1xN)@(NxK)).T = Kx1
        c_new = inv_N / (gpu_mul_Ax(self.PS, r,
                                    ngpu=ngpu, splits=splits, TYPE=self.dtype))  # (NxK)@(K,1) = N x 1
        torch.cuda.synchronize()  # just in case
        if _counter % 10 == 0:
            err = torch.sum(torch.abs((c.squeeze() / c_new.squeeze()) - ones)).cpu().item()
        c = c_new
        _counter += 1
    print("error: ", err, 'step ', _counter, flush=True)

    # getting the final tranportation matrix #####################
    for i, qq in enumerate(self.PS):
        torch.mul(qq, c[splits[i]:splits[i + 1], :].to('cuda:' + str(i + 1)), out=qq)
    [torch.mul(r.to('cuda:' + str(i + 1)).t(), qq, out=qq) for i, qq in enumerate(self.PS)]
    argmaxes = torch.empty(N, dtype=torch.int64, device='cuda:0')

    start_idx = 0
    for i, qq in enumerate(self.PS):
        amax = torch.argmax(qq, 1)
        argmaxes[start_idx:start_idx + len(qq)].copy_(amax)
        start_idx += len(qq)
    newL = argmaxes
    print('opt took {0:.2f}min, {1:4d}iters'.format(((time.time() - tt) / 60.), _counter), flush=True)
    # finally, assign the new labels ########################
    self.L[nh] = newL

