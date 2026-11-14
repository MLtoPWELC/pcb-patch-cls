"""轻量 PCB 缺陷分类骨干: 深度可分离卷积 + 空洞(膨胀)卷积 + 可选频域旁路。

设计要点(对应论文的"参数削减"叙事):
  - 全部卷积为 depthwise(3x3, 可空洞) + pointwise(1x1), 参数量约为标准卷积的 1/8~1/9
  - 用 1/2/4 的空洞率在不降分辨率的前提下扩大感受野, 避免为覆盖大缺陷再加下采样层
  - 可选 SE 通道注意力, 用极少量参数换通道重标定
  - 可选 --aux fft / hf: 频域/高频旁路, 只作为门控信号注入, 不改变主干的空域归纳偏置
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        h = max(8, c // r)
        self.fc1 = nn.Conv2d(c, h, 1)
        self.fc2 = nn.Conv2d(h, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class RepDSBlock(nn.Module):
    """深度可分离块的重参数化版本: 训练多分支, 推理合并成单分支。

    训练时 dw 位置并行三条线性通路:
        · 3x3 深度卷积(可空洞) + BN
        · 1x1 深度卷积 + BN          —— 补一个"只看单点"的视角
        · 恒等映射 + BN (仅 stride=1) —— 保留原特征, 类似残差
    三者相加后才过激活。因为卷积和 BN 都是线性的, 推理时可以把三条通路精确
    折叠成一个 3x3 深度卷积(1x1 补零到中心, 恒等视为中心为 1 的核),
    参数量和延迟回到单分支水平 —— 相当于"训练期借用容量, 推理期还给硬件"。

    dil 不改变融合方式: 3x3 用 padding=dilation, 补零后的 1x1 仍落在中心。
    """

    def __init__(self, cin, cout, stride=1, dilation=1, se=True, deploy=False):
        super().__init__()
        self.groups = cin
        self.stride = stride
        self.dilation = dilation
        self.use_id = (stride == 1)
        self.deploy = deploy
        if deploy:
            self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=dilation,
                                dilation=dilation, groups=cin, bias=True)
        else:
            self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=dilation,
                                dilation=dilation, groups=cin, bias=False)
            self.bn = nn.BatchNorm2d(cin)
            self.dw1 = nn.Conv2d(cin, cin, 1, stride=stride, groups=cin, bias=False)
            self.bn1 = nn.BatchNorm2d(cin)
            if self.use_id:
                self.bn_id = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)
        self.se = SE(cout) if se else nn.Identity()
        self.use_res = (stride == 1 and cin == cout)

    def forward(self, x):
        if self.deploy:
            y = self.act(self.dw(x))
        else:
            y = self.bn(self.dw(x)) + self.bn1(self.dw1(x))
            if self.use_id:
                y = y + self.bn_id(x)
            y = self.act(y)
        y = self.bn2(self.pw(y))
        y = self.se(y)
        if self.use_res:
            y = y + x
        return self.act(y)

    # ---- 重参数化
    @staticmethod
    def _fuse_bn(weight, bn):
        """把 BN 折叠进卷积核: w' = w * gamma/sigma, b' = beta - gamma*mu/sigma。"""
        gamma, beta = bn.weight, bn.bias
        mu, var = bn.running_mean, bn.running_var
        std = (var + bn.eps).sqrt()
        scale = (gamma / std).reshape(-1, 1, 1, 1)
        return weight * scale, beta - gamma * mu / std

    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_bn(self.dw.weight.detach(), self.bn)
        k1, b1 = self._fuse_bn(self.dw1.weight.detach(), self.bn1)
        kernel = k3.clone()
        bias = b3 + b1
        if k1.size(0) == k3.size(0) and k1.size(1) == k3.size(1):
            center = k3.size(-1) // 2
            kernel[:, :, center, center] += k1[:, :, 0, 0]
        if self.use_id:
            gamma, beta = self.bn_id.weight, self.bn_id.bias
            mu, var = self.bn_id.running_mean, self.bn_id.running_var
            std = (var + self.bn_id.eps).sqrt()
            kid = torch.zeros_like(k3)
            center = k3.size(-1) // 2
            kid[:, 0, center, center] = (gamma / std)
            kernel = kernel + kid
            bias = bias + (beta - gamma * mu / std)
        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self):
        """把三条通路合并成一个卷积, 之后 forward 走 deploy 分支。"""
        if self.deploy:
            return self
        kernel, bias = self.get_equivalent_kernel_bias()
        dw = nn.Conv2d(self.groups, self.groups, 3, stride=self.stride,
                       padding=self.dilation, dilation=self.dilation,
                       groups=self.groups, bias=True)
        # 新卷积必须落在与原权重相同的设备/精度上, 否则 GPU 上前向会报
        # "weight is on cpu, different from other tensors on cuda"
        dw = dw.to(device=kernel.device, dtype=kernel.dtype)
        dw.weight.copy_(kernel)
        dw.bias.copy_(bias)
        self.dw = dw
        for name in ("bn", "dw1", "bn1", "bn_id"):
            if hasattr(self, name):
                delattr(self, name)
        self.deploy = True
        return self


class DSBlock(nn.Module):
    """depthwise(dilated) + pointwise, stride=1 且通道一致时带残差。"""

    def __init__(self, cin, cout, stride=1, dilation=1, se=True):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=dilation,
                            dilation=dilation, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU6(inplace=True)
        self.se = SE(cout) if se else nn.Identity()
        self.use_res = (stride == 1 and cin == cout)

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        y = self.bn2(self.pw(y))
        y = self.se(y)
        if self.use_res:
            y = y + x
        return self.act(y)


class SpectralGate(nn.Module):
    """频域旁路: 对灰度图做 rFFT, 取 log 幅度谱, 池化后生成通道门控。

    aux='fft' 用真实 FFT; aux='hf' 用 原图 - 低通(原图) 的高频残差, 计算更省。
    """

    def __init__(self, channels, aux="fft"):
        super().__init__()
        self.aux = aux
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU6(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1, groups=8, bias=False), nn.BatchNorm2d(8), nn.ReLU6(inplace=True),
        )
        self.fc = nn.Linear(8, channels)

    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        if self.aux == "fft":
            spec = torch.fft.rfft2(g)
            mag = torch.log1p(spec.abs())
            feat = F.adaptive_avg_pool2d(mag, (32, 32))
        else:
            low = F.avg_pool2d(g, 5, stride=1, padding=2)
            feat = (g - low).abs()
        feat = self.conv(feat)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)
        return torch.sigmoid(self.fc(feat)).unsqueeze(-1).unsqueeze(-1)


class LightPCB(nn.Module):
    def __init__(self, num_classes=9, width=1.0, aux="none", se=True, dropout=0.1, in_ch=3,
                 dil=(1, 2, 4), depth=2, rep=False):
        """depth=2: 每级两个块(下采样 + 保持), 空洞加在保持块上(默认)。
        depth=1: 每级只留下采样块, 空洞改加在下采样块上——层数减半后感受野骤降,
                 这时空洞才真正承担扩大视野的作用, 而不是在已经够用的视野上叠冗余。
        rep=True: 用 RepDSBlock(训练多分支/推理单分支), 训练完调 switch_to_deploy() 合并。
        """
        super().__init__()
        self.rep = rep
        Block = RepDSBlock if rep else DSBlock
        w = lambda c: max(8, int(c * width))
        chs = [w(16), w(32), w(48), w(64), w(96)]
        self.stem = Block(in_ch, chs[0], stride=2, se=se)
        # depth=2 时保持块吃 dil, 下采样块 d=1; depth=1 时没有保持块, dil 移到下采样块
        d_stride, d_hold = ((1, 1, 1), dil) if depth > 1 else (dil, (1, 1, 1))
        blocks = []
        for i, (cin, cout, stride) in enumerate(
                [(chs[0], chs[1], 2), (chs[1], chs[2], 2), (chs[2], chs[3], 2)]):
            seq = [Block(cin, cout, stride=stride, dilation=d_stride[i], se=se)]
            if depth > 1:
                seq.append(Block(cout, cout, dilation=d_hold[i], se=se))
            blocks.append(nn.Sequential(*seq))
        self.s1, self.s2, self.s3 = blocks
        s4 = [Block(chs[3], chs[4], stride=1, se=se)]
        if depth > 1:
            s4.append(Block(chs[4], chs[4], dilation=1, se=se))
        self.s4 = nn.Sequential(*s4)
        self.gate = SpectralGate(chs[4], aux) if aux in ("fft", "hf") else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(chs[4], num_classes))

    @torch.no_grad()
    def switch_to_deploy(self):
        """把所有 RepDSBlock 的多分支合并成单分支; 非 rep 模型调用无副作用。"""
        if not self.rep:
            return self
        n = 0
        for m in self.modules():
            if isinstance(m, RepDSBlock):
                m.switch_to_deploy()
                n += 1
        self.rep = False
        print(f"[rep] 已合并 {n} 个 RepDSBlock")
        return self

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        if self.gate is not None:
            x = x * self.gate(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


def build_model(name, num_classes, **kw):
    name = name.lower()
    if name == "lightpcb":
        return LightPCB(num_classes, **kw)
    import torchvision.models as tvm
    if name == "resnet18":
        m = tvm.resnet18(weights=None, num_classes=num_classes)
    elif name == "mobilenetv3_small":
        m = tvm.mobilenet_v3_small(weights=None, num_classes=num_classes)
    elif name == "shufflenetv2_x0_5":
        m = tvm.shufflenet_v2_x0_5(weights=None, num_classes=num_classes)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"unknown model: {name}")
    return m


@torch.no_grad()
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_flops(model, input_size=(1, 3, 96, 96), device=None):
    """统计 MACs(乘加), 单位 M。

    探针输入必须与模型同设备: 模型已被 .to("cuda") 搬走时用 CPU 张量前向会抛
    "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)"。
    device 缺省时从模型参数推断。
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    total = {"macs": 0}

    def conv_hook(mod, inp, out):
        c_out, c_in = mod.out_channels, mod.in_channels // mod.groups
        k = mod.kernel_size[0] * mod.kernel_size[1]
        total["macs"] += out.shape[-1] * out.shape[-2] * c_out * c_in * k

    def lin_hook(mod, inp, out):
        total["macs"] += mod.in_features * mod.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    model.eval()
    model(torch.zeros(*input_size, device=device))
    for h in hs:
        h.remove()
    return total["macs"] / 1e6


@torch.no_grad()
def measure_latency(model, input_size=(1, 3, 96, 96), iters=50, warmup=10, device="cpu"):
    # 允许传入 torch.device 或字符串; 用 .type 判断才能同时兼容两者
    dev = torch.device(device)
    model.eval().to(dev)
    x = torch.randn(*input_size, device=dev)
    for _ in range(warmup):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms
