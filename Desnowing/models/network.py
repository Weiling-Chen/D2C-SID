import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent, Normal
import os
from PIL import Image
from torchvision.transforms import ToTensor


# 先验后验
class Prponet(nn.Module):
    def __init__(self, latent_dim):
        super(Prponet, self).__init__()
        self.latent_dim = latent_dim
        self.pr_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
            # nn.AdaptiveAvgPool2d((64, 64))
        )
        self.po_encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        self.fc_mu = nn.Linear(64*64*64, self.latent_dim) ####改!!!!!!
        self.fc_logvar = nn.Linear(64*64*64, self.latent_dim) ####改
        self.fc1 = nn.Linear(self.latent_dim, 4096)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y, po=True):
        if po:
            a = self.po_encoder(torch.cat((x, y), dim=1))

            a1 = torch.isnan(a).any().item()

            if a1:
                print("a1 中存在 NaN 值！")

            a = a.view(-1, 64*64*64)

            mu = self.fc_mu(a)
            logvar = self.fc_logvar(a)

            dist_po = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

            # 重参数化
            # z = self.reparameterize(mu, logvar)
            z = dist_po.rsample()
            # z = mu + -15*torch.exp(logvar)
            z = self.fc1(z)
            z = z.view(-1, 1, 64, 64)

            return z, mu, logvar, dist_po
        else:
            x = self.pr_encoder(x)
            x = x.view(-1, 64 * 64 * 64)  ####改
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            dist_pr = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)

            # 重参数化
            # z = self.reparameterize(mu, logvar)
            z = dist_pr.rsample()
            # z = mu + -15*torch.exp(logvar)
            z = self.fc1(z)
            z = z.view(-1, 1, 64, 64)

            return z, mu, logvar, dist_pr


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# 经过ShallowLayer之后，图像的hw不会改变，但是通道变了
class ShallowLayer(nn.Module):
    def __init__(self, out_plane):
        super(ShallowLayer, self).__init__()
        self.main = nn.Sequential(
            BasicConv(1, out_plane // 4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x


# --- Channel Attention (CA) Layer --- #
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.channel_attn = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.channel_attn(y)
        return x * y 


class SALayer(nn.Module):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.spatial_attn = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
                nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        h = torch.cat([avg_out, max_out], dim=1)
        y = self.spatial_attn(h)

        return x * y


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 512)
        self.fc2 = nn.Linear(512, 768)

        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(64)
        )
        self.up1 = nn.ConvTranspose2d(in_channels=64,
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(67, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.PReLU(512)
        )
        self.up2 = nn.ConvTranspose2d(in_channels=512,
                                      out_channels=512,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(515, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(128)
        )
        self.up3 = nn.ConvTranspose2d(in_channels=128,
                                      out_channels=128,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(131, 64, kernel_size=3, stride=1, padding=1),
            nn.PReLU(64),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.PReLU(32)
        )
        self.up4 = nn.ConvTranspose2d(in_channels=32,
                                      out_channels=32,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)

        self.layer5 = nn.Sequential(
            nn.Conv2d(35, 128, kernel_size=3, stride=1, padding=1),
            nn.PReLU(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.PReLU(256),
            nn.Conv2d(256, 3, kernel_size=3, stride=1, padding=1)
            )
            
        self.attn_layer1 = CALayer(channel=64)
        self.attn_layer2 = CALayer(channel=512)
        self.attn_layer3 = CALayer(channel=128)
        self.attn_layer4 = CALayer(channel=32)

        self.tanh = nn.Tanh()
        

    def forward(self, img, mean, var):
        _, _, H, W = img.size()
        std = torch.exp(var / 2)
        eps = torch.randn_like(std)
        z = mean + eps * std

        # decode
        x = self.fc1(z)
        x = self.fc2(x)
        x = x.view(-1, 3, 16, 16)
        hidden_map = x
        condition = F.interpolate(img, size=(H//16, W//16), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer1(x)
        x = self.attn_layer1(x)
        x = self.up1(x)
        condition = F.interpolate(img, size=(H//8, W//8), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer2(x)
        x = self.attn_layer2(x)
        x = self.up2(x)
        condition = F.interpolate(img, size=(H//4, W//4), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer3(x)
        x = self.attn_layer3(x)
        x = self.up3(x)
        condition = F.interpolate(img, size=(H//2, W//2), mode='bicubic', align_corners=True)
        x = torch.cat((x, condition), 1)
        
        x = self.layer4(x)
        x = self.attn_layer4(x)
        x = self.up4(x)
        x = torch.cat((x, img), 1)

        output = self.layer5(x)
        output = self.tanh(output)
        
        return output, hidden_map

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VisualTransformer(image_width=256,
                                         image_height=256,
                                         patch_size=16,
                                         num_dim=50)
        self.decoder = Decoder()
         
    def en(self, x):
        return self.encoder(x)
    
    def de(self, x, mean, log_var):
        return self.decoder(x, mean, log_var)
        
    def forward(self, x):
        mean, log_var = self.en(x)
        output, hidden_map = self.de(x, mean, log_var)
        output = torch.add(x, output)

        return output, mean, log_var
if __name__ == '__main__':

    folder_path1 = r"D:\lfq\FSNet-main\Desnowing\dataset\train\GT"
    image_tensors1 = []
    for file_name in os.listdir(folder_path1):
        if file_name.endswith(('.bmp', '.jpeg', '.png')):
            image_path1 = os.path.join(folder_path1, file_name)
            image1 = Image.open(image_path1).convert('L')
            image_tensor1 = ToTensor()(image1)
            image_tensor1 = image_tensor1.unsqueeze(0)
            image_tensors1.append(image_tensor1)

    folder_path12 = r"D:\lfq\FSNet-main\Desnowing\dataset\train\Snow"
    image_tensors12 = []
    for file_name in os.listdir(folder_path1):
        if file_name.endswith(('.bmp', '.jpeg', '.png')):
            image_path12 = os.path.join(folder_path12, file_name)
            image12 = Image.open(image_path12).convert('L')
            image_tensor12 = ToTensor()(image12)
            image_tensor12 = image_tensor12.unsqueeze(0)
            image_tensors12.append(image_tensor12)

    model = Prponet(10)
    for i, image_tensor in enumerate(image_tensors1):
        output = model(image_tensor, image_tensor)
        if output is not None:
            print(f"图片 {i} 不存在 nan 值")