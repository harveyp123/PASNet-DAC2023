import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import logging

from models.gate_tmp import PolyAct, GateAct, GatePool

__dict__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
]

# -------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        act_type, 
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.act_type1 = eval(act_type+"()")
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.act_type2 = eval(act_type+"()")
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_type1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act_type2(out)
        # out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        act_type, 
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.act_type1 = eval(act_type+"()")
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.act_type2 = eval(act_type+"()")
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.act_type3 = eval(act_type+"()")
        # self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_type1(out)
        # out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_type2(out)
        # out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act_type3(out)
        # out = self.relu(out)

        return out


class ResNet_Gated(nn.Module):
    def __init__(
        self,
        block,
        layers,
        criterion, 
        act_type="nn.ReLU", 
        pool_type="nn.MaxPool2d",
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet_Gated, self).__init__()
        
        self.criterion = criterion

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.bn1 = norm_layer(self.inplanes)

        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act_type = eval(act_type+"()")
        self.pool_type = eval(pool_type + "(kernel_size=3, stride=2, padding=1)")

        self.layer1 = self._make_layer(block, 64, layers[0], act_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], act_type, stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], act_type, stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], act_type, stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        ### Get the alphas, poly function data, and weight data
        self._alphas = []
        self._poly = []
        self._weights = []
        self._lat = []
        for name, parameter in self.named_parameters():
            if 'alpha' in name:
                self._alphas.append((name, parameter))                
            elif 'lat' in name: 
                self._lat.append((name, parameter))   
            else: 
                self._weights.append((name, parameter))
            if 'poly' in name:
                self._poly.append((name, parameter))
        self._gate_act_name = []
        self._gate_act_model = []
        self._gate_pool_name = []
        self._gate_pool_model = []
        for name, model_stat in self.named_modules(): 
            if type(model_stat).__name__ == 'GatePool':
                self._gate_pool_name.append(name)
                self._gate_pool_model.append(model_stat)
            if type(model_stat).__name__ == 'GateAct':
                self._gate_act_name.append(name)
                self._gate_act_model.append(model_stat)

    def _make_layer(self, block, planes, blocks, act_type, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                act_type, 
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    act_type, 
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_type(x)
        x = self.pool_type(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def loss_lat(self, lat_lamda):
        lat_loss = []
        sftmx = torch.nn.Softmax(dim=0)
        for index, (name, alpha_para) in enumerate(self._alphas):
            c = sftmx(alpha_para)
            lat_loss.append(c[0] * self._lat[index][1][0] + c[1] * self._lat[index][1][1])
        loss_latency = lat_lamda * torch.norm(torch.stack(lat_loss),1)
        return loss_latency

    def weights(self):
        for n, p in self._weights:
            yield p
    def named_weights(self):
        for n, p in self._weights:
            yield n, p
    def alphas(self):
        for n, p in self._alphas:
            yield p
    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
    def poly(self):
        for n, p in self._poly:
            yield p
    def named_poly(self):
        for n, p in self._poly:
            yield n, p

    def load_pretrained(self, pretrained_path = False):
        if pretrained_path:
            if os.path.isfile(pretrained_path):
                print("=> loading checkpoint '{}'".format(pretrained_path))
                checkpoint = torch.load(pretrained_path)                
                # pretrained_dict = checkpoint['state_dict']
                pretrained_dict = checkpoint
                model_dict = self.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict) 
                self.load_state_dict(model_dict)
            else:
                print("=> no checkpoint found at '{}'".format(pretrained_path))

    def load_NAS_alpha(self, pretrained_path = False):
        if pretrained_path:
            if os.path.isfile(pretrained_path):
                print("=> loading NAS search alpha from '{}'".format(pretrained_path))
                checkpoint = torch.load(pretrained_path)
                with torch.no_grad():
                    for index, alphas in enumerate(self._alphas):
                        alphas[1].copy_(checkpoint._alphas[index][1])
                del checkpoint
            else:
                print("=> no checkpoint found at '{}'".format(pretrained_path))

    def load_check_point(self, check_point_path = False):
        if os.path.isfile(check_point_path):
            print("=> loading checkpoint from '{}'".format(check_point_path))
            checkpoint = torch.load(check_point_path)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            self.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint at epoch {}"
                  .format(checkpoint['epoch']))
            print("Is best result?: ", checkpoint['is_best'])
            return start_epoch, best_prec1
        else:
            print("=> no checkpoint found at '{}'".format(check_point_path))
               
    def save_checkpoint(self, epoch, best_prec1, is_best, filename='checkpoint.pth.tar'):
        """
        Save the training model
        """
        state = {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'best_prec1': best_prec1,
                'is_best': is_best
            }
        torch.save(state, filename)

    def change_gate(self, Gate_status = False):
        """
        Save the model gate
        """
        for gate_act_model in self._gate_act_model:
            gate_act_model.gate = Gate_status
        for gate_pool_model in self._gate_pool_model:
            gate_pool_model.gate = Gate_status

    def change_scaling(self, weight_scale = 1):
        """
        Save the model gate
        """
        print("change scaling to: ", weight_scale)
        for gate_act_model in self._gate_act_model:
            gate_act_model.weight_scale = weight_scale

    def set_alpha_all_poly_avgpool(self, set_alpha = True):
        """
        Set the model alpha manually
        """
        if set_alpha:
            with torch.no_grad():
                alpha_set = nn.Parameter(data=torch.Tensor([0.0, 1.0]), requires_grad=True)
                for alphas in self._alphas:
                    alphas[1].copy_(alpha_set)
        else:
            with torch.no_grad():
                alpha_set = nn.Parameter(data=torch.Tensor([1.0, 0.0]), requires_grad=True)
                for alphas in self._alphas:
                    alphas[1].copy_(alpha_set)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        # Logging alpha data: 
        logger.info("####### ALPHA #######")
        logger.info("# Alpha for the model")
        for alpha_name, alpha_para in self.named_alphas():
            logger.info([alpha_name, F.softmax(alpha_para, dim=-1)])
        logger.info("#####################")
        
        # Logging Poly layer data: 
        logger.info("####### ALPHA #######")
        logger.info("# Alpha for the model")
        for poly_name, poly_para in self.named_poly():
            logger.info([poly_name, poly_para])
        logger.info("#####################")
        
        # restore formats
        for handler, formatter in zip(logger.handlers, org_formatters):
            handler.setFormatter(formatter)
        
# ------------------------------------------------------

def _resnet(arch, block, layers, pretrained, progress, device, criterion, act_type, pool_type, **kwargs):
    model = ResNet_Gated(block, layers, criterion, act_type=act_type, pool_type=pool_type, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/" + arch + ".pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(criterion, act_type, pool_type, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
                "resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, device, 
                criterion, act_type, pool_type, **kwargs
            )


def resnet34(criterion, act_type, pool_type, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
                "resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, device, 
                criterion, act_type, pool_type, **kwargs
            )


def resnet50(criterion, act_type, pool_type, pretrained=False, progress=True, device="cpu", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
                "resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, device, 
                criterion, act_type, pool_type, **kwargs
            )
