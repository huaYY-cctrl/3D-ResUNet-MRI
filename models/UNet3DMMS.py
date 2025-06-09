import torch
import torch.nn as nn
from torchinfo import summary


class Conv3D_Block(nn.Module):
    def __init__(self, in_feat, out_feat, kernel=3, stride=1, padding=1, residual=True):
        """
        3D卷积模块，可选择是否包含残差连接

        参数:
            in_feat (int): 输入特征图的通道数
            out_feat (int): 输出特征图的通道数
            kernel (int): 卷积核大小，默认3
            stride (int): 步长，默认1
            padding (int): 填充大小，默认1
            residual (bool): 是否使用残差连接，默认True
        """
        super(Conv3D_Block, self).__init__()

        # 定义主路径的两层3D卷积结构
        self.conv = nn.Sequential(
            # 第一层卷积 + BN + ReLU
            nn.Conv3d(in_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),

            # 第二层卷积 + BN + ReLU
            nn.Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True)
        )

        # 残差连接标志
        self.residual = residual

        # 如果启用残差连接，需要定义1x1x1卷积来匹配维度
        if self.residual:
            self.residual_conv = nn.Conv3d(in_feat, out_feat, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量 [B, C, D, H, W]

        返回:
            torch.Tensor: 输出张量
        """
        # 保存输入作为残差
        res = x

        # 如果启用残差连接，返回卷积结果 + 残差连接
        if self.residual:
            return self.conv(x) + self.residual_conv(res)
        else:
            # 否则只返回卷积结果
            return self.conv(x)


class Up_Block(nn.Module):
    def __init__(self, init_feat, scale_factor=(2, 2, 2)):
        """
        3D上采样模块，用于UNet架构中的解码器部分

        参数:
            init_feat (int): 输入特征图的通道数
            scale_factor (tuple): 上采样的缩放因子，默认(2, 2, 2)表示在深度、高度和宽度上都放大2倍
        """
        super(Up_Block, self).__init__()

        # 定义3D三线性上采样层，使用align_corners=True保持角点对齐
        # align_corners=True 更适合需要严格保持几何形状的任务（如医学图像）边界区域的精确性至关重要
        self.up = nn.Upsample(scale_factor=scale_factor, mode="trilinear", align_corners=True)

        # 定义3x3卷积层，将通道数减半
        self.conv = nn.Conv3d(init_feat, int(init_feat / 2), kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量 [B, C, D, H, W]

        返回:
            torch.Tensor: 输出张量 [B, C/2, D*2, H*2, W*2]
        """
        # 执行上采样操作，增大空间维度
        out = self.up(x)

        # 通过卷积减少通道数，同时提取特征
        out = self.conv(out)

        return out


class UNet3DMMS(nn.Module):
    def __init__(self, input_ch=1, output_ch=4, init_feats=16):
        """
        多尺度3D UNet模型，专为心脏MRI分割设计

        参数:
            input_ch (int): 输入通道数，默认1（灰度MRI）
            output_ch (int): 输出通道数，默认4（对应不同心脏结构类别）
            init_feats (int): 初始特征通道数，默认16
        """
        super(UNet3DMMS, self).__init__()

        # 编码器部分：使用不同kernel_size的MaxPool3d实现多尺度下采样
        # 采用非对称池化策略，更好地处理3D医学图像的各向异性特性
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 仅在H/W维度下采样
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 在D/H/W维度下采样
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 解码器部分：使用Up_Block实现上采样，逐步恢复空间分辨率
        self.up7 = Up_Block(init_feat=init_feats * 32, scale_factor=(1, 2, 2))
        self.up8 = Up_Block(init_feat=init_feats * 16, scale_factor=(2, 2, 2))
        self.up9 = Up_Block(init_feat=init_feats * 8, scale_factor=(1, 2, 2))
        self.up10 = Up_Block(init_feat=init_feats * 4, scale_factor=(2, 2, 2))
        self.up11 = Up_Block(init_feat=init_feats * 2, scale_factor=(1, 2, 2))

        # 卷积块：使用带残差连接的3D卷积块(Conv3D_Block)增强特征提取能力
        self.conv1 = Conv3D_Block(in_feat=input_ch, out_feat=init_feats)
        self.conv2 = Conv3D_Block(in_feat=init_feats, out_feat=init_feats * 2)
        self.conv3 = Conv3D_Block(in_feat=init_feats * 2, out_feat=init_feats * 4)
        self.conv4 = Conv3D_Block(in_feat=init_feats * 4, out_feat=init_feats * 8)
        self.conv5 = Conv3D_Block(in_feat=init_feats * 8, out_feat=init_feats * 16)
        self.conv6 = Conv3D_Block(in_feat=init_feats * 16, out_feat=init_feats * 32)  # 瓶颈层

        # 解码器卷积块
        self.conv7 = Conv3D_Block(in_feat=init_feats * 32, out_feat=init_feats * 16)
        self.conv8 = Conv3D_Block(in_feat=init_feats * 16, out_feat=init_feats * 8)
        self.conv9 = Conv3D_Block(in_feat=init_feats * 8, out_feat=init_feats * 4)
        self.conv10 = Conv3D_Block(in_feat=init_feats * 4, out_feat=init_feats * 2)
        self.conv11 = Conv3D_Block(in_feat=init_feats * 2, out_feat=init_feats)

        # 最终1x1卷积层：将特征映射转换为类别预测
        self.conv12 = nn.Conv3d(init_feats, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量 [B, C, D, H, W]

        返回:
            torch.Tensor: 输出分割结果 [B, num_classes, D, H, W]
        """
        # 编码器路径：特征提取与下采样
        conv1 = self.conv1(x)  # 第一次卷积，保留原始分辨率特征
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        pool5 = self.pool5(conv5)

        conv6 = self.conv6(pool5)  # 瓶颈层，捕获高级抽象特征

        # 解码器路径：上采样与特征融合（跳跃连接）
        up7 = self.up7(conv6)
        conv7 = self.conv7(torch.cat([conv5, up7], dim=1))  # 融合编码器和解码器特征

        up8 = self.up8(conv7)
        conv8 = self.conv8(torch.cat([conv4, up8], dim=1))

        up9 = self.up9(conv8)
        conv9 = self.conv9(torch.cat([conv3, up9], dim=1))

        up10 = self.up10(conv9)
        conv10 = self.conv10(torch.cat([conv2, up10], dim=1))

        up11 = self.up11(conv10)
        conv11 = self.conv11(torch.cat([conv1, up11], dim=1))

        # 最终分类层：将特征映射转换为类别预测
        conv12 = self.conv12(conv11)

        return conv12


if __name__ == '__main__':
    device = torch.device("cpu")
    model = UNet3DMMS(1, 4).to(device)
    summary(model, (1, 1, 16, 128, 128))
