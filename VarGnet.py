import torch
import torch.nn.functional as F
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=True):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class separable_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel = 3,  stride=1, pad = 1, 
                     factor=1,
                     bias=False,
                     bn_dw_out=True,
                     act_dw_out=True,
                     bn_pw_out=True,
                     act_pw_out=True,
                     dilate=1):
        super(separable_conv2d, self).__init__()
        self.bn_dw_out = bn_dw_out
        self.bn_pw_out = bn_pw_out
        self.act_pw_out = act_pw_out
        self.act_dw_out = act_dw_out
        expansion = int(in_planes* factor)
        self.conv1 = nn.Conv2d(in_planes, expansion, kernel_size=kernel, stride=stride,
                     padding=pad, bias=bias, groups = int(in_planes / 8))
        
        self.conv2 = nn.Conv2d(expansion, out_planes, kernel_size = 1, stride=1,
                     padding=0, bias= bias, groups = 1)
        if bn_dw_out:
            self.bn1 = nn.BatchNorm2d(expansion)

        if bn_pw_out:
            self.bn2 = nn.BatchNorm2d(out_planes)


    def forward(self, x):
        out = self.conv1(x)
        if self.bn_dw_out:
            out = self.bn1(out)
        if self.act_dw_out:
            out = F.relu(out)
        out = self.conv2(out)
        if self.bn_pw_out:
            out = self.bn2(out)
        if self.act_pw_out:
            out = F.relu(out)

        return out
class vargnet_block(nn.Module):
    """docstring for vargnet_block"""
    def __init__(self,
                    in_planes,
                    n_out_ch,
                  factor=2,
                  multiplier=1,
                  kernel=3,
                  stride=1,
                  dilate=1,
                  with_dilate=False, dim_match = True, 
                  use_se = True):
        super(vargnet_block, self).__init__()
        out_planes_1 = int(n_out_ch[0] * multiplier)
        out_planes_2 = int(n_out_ch[1] * multiplier)
        out_planes_3 = int(n_out_ch[2] * multiplier)
        pad = (((kernel[0] - 1) * dilate + 1) // 2,
           ((kernel[1] - 1) * dilate + 1) // 2)
        if with_dilate:
            stride = 1
        self.dim_match = dim_match
        if dim_match:
            pass
        else:
            self.conv = separable_conv2d(out_planes_1, out_planes_3, kernel = kernel, stride=stride, pad = pad, factor=factor,
                                     bias=False,
                                     act_pw_out=False,
                                     dilate= dilate )

        self.sep1 = separable_conv2d(
                                 in_planes=out_planes_1,
                                 out_planes=out_planes_2,
                                 kernel=kernel,
                                 pad=pad,
                                 stride=stride,
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate)
        self.sep2 = separable_conv2d(in_planes=out_planes_2,
                                 out_planes=out_planes_3,
                                 kernel=kernel,
                                 pad=pad,
                                 stride=1,
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate,
                                 act_pw_out=False )
        self.use_se = use_se
        if use_se:
            self.se = SEModule(out_planes_3)
    def forward(self, x):
        if self.dim_match:
            short_cut = x 
        else:
            short_cut = self.conv(x)
        out = self.sep1(x)
        out = self.sep2(out)
        if self.use_se:
            out = self.se(out)
        out_data = out + short_cut
        out_data = F.relu(out_data)
        return out_data
class vargnet_branch_merge_block(nn.Module):
    def __init__(self, in_planes, 
                     n_out_ch,
                               factor=2,
                               dim_match=False,
                               multiplier=1,
                               kernel=3,
                               stride=2,
                               dilate=1,
                               with_dilate=False,
                               name=None):
        super(vargnet_branch_merge_block, self).__init__()
        out_planes_1 = int(n_out_ch[0] * multiplier)
        out_planes_2 = int(n_out_ch[1] * multiplier)
        out_planes_3 = int(n_out_ch[2] * multiplier)
        pad = (((kernel[0] - 1) * dilate + 1) // 2,
           ((kernel[1] - 1) * dilate + 1) // 2)
        if with_dilate:
            stride = 1
        self.dim_match = dim_match
        if dim_match:
            short_cut = data
        else:
            self.conv = separable_conv2d(out_planes_1, out_planes_3, kernel = kernel, pad = pad, stride=stride, factor=factor,
                                     bias=False,
                                     act_pw_out=False,
                                     dilate= dilate )

        self.sep1_branch1 = separable_conv2d( in_planes=out_planes_1,
                                         out_planes=out_planes_2,
                                         kernel=kernel,
                                         pad=pad,
                                         stride=stride,
                                         factor=factor,
                                         bias=False,
                                         dilate=dilate)
        self.sep1_branch2 = separable_conv2d(in_planes=out_planes_1,
                                 out_planes=out_planes_2,
                                 kernel=kernel,
                                 pad=pad,
                                 stride=stride,
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate,
                                 act_pw_out=False )
        self.sep2_branch = separable_conv2d(in_planes=out_planes_2,
                                 out_planes=out_planes_3,
                                 kernel=kernel,
                                 pad=pad,
                                 stride=1,
                                 factor=factor,
                                 bias=False,
                                 dilate=dilate,
                                 act_pw_out=False )
    def forward(self, x):
        if self.dim_match:
            short_cut = x 
        else:
            short_cut = self.conv(x)
        out1_branch1 = self.sep1_branch1(x)
        out1_branch2 = self.sep1_branch2(x)
        out1 = out1_branch1 + out1_branch2
        out1 = F.relu(out1)
        out2_branch = self.sep2_branch(out1)
        out_data = out2_branch + short_cut
        out_data = F.relu(out_data)
        return out_data

class SEModule(nn.Module):

    def __init__(self, channels, reduction = 4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
        
class head_block(nn.Module):
    def __init__(self, in_planes, out_planes,  multiplier,
                   head_pooling=False,
                   kernel=3,
                   stride=1,
                   pad=1):
        super(head_block, self).__init__()
        channels = int(out_planes * multiplier)
        self.conv1 = conv3x3(in_planes, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.head_pooling = head_pooling
        if head_pooling:
            pass
        else:
            self.last_nn = vargnet_block(channels, 
                                      [out_planes, out_planes, out_planes],
                                      factor=1,
                                      dim_match=False,
                                      multiplier=multiplier,
                                      kernel=(kernel, kernel),
                                      stride=2,
                                      dilate=1,
                                      with_dilate=False, 
                                     )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.last_nn(out)

        return out
class vargnet_body_block(nn.Module):
    """docstring for ClassName"""
    def __init__(self, in_planes, stage,
                           units,
                           out_planes,
                           kernel=(3, 3),
                           stride=(2, 2),
                           multiplier=1,
                           factor=2,
                           dilate=1,
                           with_dilate=False,
                           name=None):
        super(vargnet_body_block, self).__init__()
        assert stage >= 2, 'stage is {}, stage must be set >=2'.format(stage)
        self.branch_merge_block = vargnet_branch_merge_block(in_planes=in_planes,
                                          n_out_ch = [in_planes, out_planes, out_planes],
                                          factor=factor,
                                          dim_match=False,
                                          multiplier=multiplier,
                                          kernel=kernel,
                                          stride=stride,
                                          dilate=dilate,
                                          with_dilate=with_dilate,
                                          )
        self.units = units
        for i in range(units - 1):
            block = vargnet_block(in_planes= out_planes,
                                 n_out_ch = [out_planes, out_planes, out_planes],
                                 factor=factor,
                                 dim_match=True,
                                 multiplier=multiplier,
                                 kernel=kernel,
                                 stride=1,
                                 dilate=dilate,
                                 with_dilate=with_dilate )
            self.__setattr__("varg_block_" + str(i), block)
    def forward(self, x):
        x = self.branch_merge_block(x)
        for i in range(self.units - 1):
            x = self.__getattr__("varg_block_" + str(i))(x)
        return x
class VargNet(nn.Module):
    """docstring for VargNet"""
    def __init__(self, multiplier = 1.25):
        super(VargNet, self).__init__()
        self.num_stage = 3
        stage_list = [2, 3, 4]
        units = [3, 7, 4]
        filter_list = [32, 64, 128, 256]
        dilate_list = [1, 1, 1]
        with_dilate_list = [False, False, False]
        self.head_block = head_block(3, 32, multiplier = multiplier, head_pooling = False, kernel=3, stride=1, pad=1)
        in_planes = 32
        for i in range(self.num_stage):
            stage_block = vargnet_body_block(in_planes = in_planes, stage=stage_list[i],
                                      units=units[i],
                                      out_planes = filter_list[i + 1],
                                      kernel=(3,3),
                                      stride=2,
                                      multiplier=multiplier,
                                      factor=2,
                                      dilate=dilate_list[i],
                                      with_dilate=with_dilate_list[i])
            self.__setattr__("stage_" + str(i), stage_block)
            in_planes = filter_list[i + 1]
        if in_planes != 1024:
            self.last_conv = nn.Sequential(nn.Conv2d(320, 1024, kernel_size=1,
                             padding=0, stride = 1),
                             nn.BatchNorm2d(1024),
                             nn.ReLU())
        self.convx_depthwise = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=7,
                             padding=0, stride = 1, groups = int(1024 / 8)),
                             nn.BatchNorm2d(1024))
        self.last_pool = nn.AvgPool2d(7, stride=1)
        # pointwise
        self.convx_pointwise = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1,
                             padding=0, stride = 1),
                             nn.BatchNorm2d(512), 
                             nn.ReLU())

    def forward(self, x):
        x = self.head_block(x)
        for i in range(self.num_stage):
            x = self.__getattr__("stage_" + str(i))(x)
        x= self.last_conv(x)
        # x = self.last_pool(x)
        x = self.convx_depthwise(x)
        x = self.convx_pointwise(x)
        return x
import time
from torchsummary import summary
if __name__ == "__main__":
    model = VargNet().cuda()
    
    # model = EfficientNet().cuda()
    model.eval()
    test_data = torch.rand(32, 3, 112, 112).cuda()
    summary(model, (3,112, 112))
    for i in range(5):
        t = time.time()
        test_outputs = model(test_data) #, test_data_2]
        t2 = time.time()
        print(t2 -t)
        t = t2

        