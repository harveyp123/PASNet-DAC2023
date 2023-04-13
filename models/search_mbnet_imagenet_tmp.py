import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import logging

from models.gate_tmp import PolyAct, GateAct, GatePool

__dict__ = ["MobileNetV2", "mobilenet_v2"]


class ConvBNAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, act_type, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNAct, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            # nn.ReLU6(inplace=True),
            eval(act_type+"()")
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, act_type, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.act_type = act_type
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNAct(inp, hidden_dim, act_type, kernel_size=1))
        layers.extend(
            [
                # dw
                ConvBNAct(hidden_dim, hidden_dim, act_type, stride=stride, groups=hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_Gated(nn.Module):
    def __init__(self, criterion, act_type, num_classes=1000, width_mult=1.0):
        super(MobileNetV2_Gated, self).__init__()
        self.criterion = criterion

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        # CIFAR10
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],  # Stride 2 -> 1 for CIFAR-10
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # END

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        # CIFAR10: stride 2 -> 1
        features = [ConvBNAct(3, input_channel, act_type, stride=2)]
        # END

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, act_type, stride, expand_ratio=t)
                )
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNAct(input_channel, self.last_channel, act_type, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

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
                self._gate_act_name.append(name)
                self._gate_act_model.append(model_stat)
            if type(model_stat).__name__ == 'GateAct':
                self._gate_act_name.append(name)
                self._gate_act_model.append(model_stat)
                
    def forward(self, x):
        x = self.features(x)
        # x = x.mean([2, 3])
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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



def mobilenet_v2(criterion, act_type, pretrained=False, progress=True, device="cpu", **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2_Gated(criterion, act_type, **kwargs)
    if pretrained:
        script_dir = os.path.dirname(__file__)
        state_dict = torch.load(
            script_dir + "/state_dicts/mobilenet_v2.pt", map_location=device
        )
        model.load_state_dict(state_dict)
    return model
