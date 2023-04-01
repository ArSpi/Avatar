import math

import numpy as np
import torch
import tinycudann as tcnn

from activation import trunc_exp


class HashGridNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_layers = self.cfg.models.num_layers
        self.hidden_dim = self.cfg.models.hidden_dim
        self.geo_feat_dim = self.cfg.models.geo_feat_dim
        self.num_layers_color = self.cfg.models.num_layers_color
        self.hidden_dim_color = self.cfg.models.hidden_dim_color
        self.bound = self.cfg.renderer.bound
        self.expression_dim = self.cfg.models.expression_dim
        self.latent_code_dim = self.cfg.models.latent_code_dim

        per_level_scale = np.exp2(np.log2(2048 * self.bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,  # 分辨率的级数
                "n_features_per_level": 2,  # 在每个分辨率哈希表条目中的特征向量的维数
                "log2_hashmap_size": 19,  # 哈希表的大小（以2取对数的结果）
                "base_resolution": 16,  # 最粗糙的分辨率的大小（每一维的长度）
                "per_level_scale": per_level_scale,  # 每级分辨率网格相较于上一级分辨率网格扩大的尺度
            }
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32 + self.expression_dim + self.latent_code_dim,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",  # 隐藏层的激活函数
                "output_activation": "None",  # 输出层的激活函数
                "n_neurons": self.hidden_dim,  # 隐藏层的维数
                "n_hidden_layers": self.num_layers,  # 隐藏层的数目
            }
        )

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,  # 使用SH的最高阶数，共使用degree^2个球谐系数
            }
        )

        self.color_net = tcnn.Network(
            n_input_dims=self.encoder_dir.n_output_dims + self.geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color,
            }
        )

    def forward(self, x, d, exp, latent_code):
        # ----- density ----- #
        x = (x + self.bound) / (2 * self.bound)
        x = self.encoder(x)

        batch = x.shape[0]
        batch_exp = torch.broadcast_to(exp, (batch,) + exp.shape)
        batch_latent_code = torch.broadcast_to(latent_code, (batch,) + latent_code.shape)

        x = torch.cat((x, batch_exp, batch_latent_code), dim=-1)
        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # ----- color ----- #
        d = (d + 1) / 2
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x, exp, latent_code):
        x = (x + self.bound) / (2 * self.bound)  # to [0, 1]
        x = self.encoder(x)

        batch = x.shape[0]
        batch_exp = torch.broadcast_to(exp, (batch,) + exp.shape)
        batch_latent_code = torch.broadcast_to(latent_code, (batch,) + latent_code.shape)

        x = torch.cat((x, batch_exp, batch_latent_code), dim=-1)
        h = self.sigma_net(x)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def get_params(self, dataset, lr, trainable_background, use_latent_codes):
        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}
        ]
        if trainable_background:
            params.append({'params': dataset.background, 'lr': lr})
        if use_latent_codes:
            params.append({'params': dataset.latent_codes, 'lr': lr})
        return params





class ConditionalBlendshapePaperNeRFModel(torch.nn.Module):
    r"""Implements the NeRF model as described in Fig. 7 (appendix) of the
    arXiv submission (v0). """

    def __init__(
            self,
            num_layers=8,  # 4
            hidden_size=256,  # 256
            skip_connect_every=4,
            num_encoding_fn_xyz=6,  # 10
            num_encoding_fn_dir=4,  # 4
            include_input_xyz=True,  # True
            include_input_dir=True,  # False
            include_expression=True,
            latent_code_dim=32

    ):
        super(ConditionalBlendshapePaperNeRFModel, self).__init__()

        include_input_xyz = 3 if include_input_xyz else 0  # 3
        include_input_dir = 3 if include_input_dir else 0  # 0
        include_expression = 76 if include_expression else 0  # 76

        self.dim_xyz = include_input_xyz + 2 * 3 * num_encoding_fn_xyz  # 63
        self.dim_dir = include_input_dir + 2 * 3 * num_encoding_fn_dir  # 27
        self.dim_expression = include_expression  # 76
        self.dim_latent_code = latent_code_dim  # 32

        self.layers_xyz = torch.nn.ModuleList()
        self.layers_xyz.append(torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code, 256))
        for i in range(1, 6):
            if i == 3:
                self.layers_xyz.append(
                    torch.nn.Linear(self.dim_xyz + self.dim_expression + self.dim_latent_code + 256, 256))
            else:
                self.layers_xyz.append(torch.nn.Linear(256, 256))
        self.fc_feat = torch.nn.Linear(256, 256)
        self.fc_alpha = torch.nn.Linear(256, 1)

        self.layers_dir = torch.nn.ModuleList()
        self.layers_dir.append(torch.nn.Linear(256 + self.dim_dir, 128))
        for i in range(3):
            self.layers_dir.append(torch.nn.Linear(128, 128))
        self.fc_rgb = torch.nn.Linear(128, 3)
        self.relu = torch.nn.functional.relu

    '''
    initial(63+76+32) ==> Linear(63+76+32 -> 256) ==> relu ==> x(256)
    x(256) ==> Linear(256 -> 256) ==> relu ==> x(256)
    x(256) ==> Linear(256 -> 256) ==> relu ==> x(256)
    x(256) & initial(63+76+32) ==> Linear(256+63+76+32 -> 256) ==> relu ==> x(256)
    x(256) ==> Linear(256 -> 256) ==> relu ==> x(256)
    x(256) ==> Linear(256 -> 256) ==> relu ==> x(256)

    x_(256) ==> Linear(256 -> 256) ==> feat(256)
    feat(256) ==> Linear(256 -> 1) ==> alpha(1) ### 获得alpha

    feat(256) & dirs(27) ==> Linear(256+27 -> 128) ==> relu ==> x(128)
    x(128) ==> Linear(128 -> 128) ==> relu ==> x(128)
    x(128) ==> Linear(128 -> 128) ==> relu ==> x(128)

    x(128) ==> Linear(128 -> 3) ==> rgb(3) ### 获得RGB
    '''

    def forward(self, x, expr=None, latent_code=None, **kwargs):
        xyz, dirs = x[..., : self.dim_xyz], x[..., self.dim_xyz:]
        x = xyz  # self.relu(self.layers_xyz[0](xyz))
        latent_code = latent_code.repeat(xyz.shape[0], 1)
        if self.dim_expression > 0:
            expr_encoding = (expr * 1 / 3).repeat(xyz.shape[0], 1)
            initial = torch.cat((xyz, expr_encoding, latent_code), dim=1)
            x = initial
        for i in range(6):
            if i == 3:
                x = self.layers_xyz[i](torch.cat((initial, x), -1))
            else:
                x = self.layers_xyz[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        alpha = self.fc_alpha(feat)
        x = self.layers_dir[0](torch.cat((feat, dirs), -1))
        x = self.relu(x)
        for i in range(1, 3):
            x = self.layers_dir[i](x)
            x = self.relu(x)
        rgb = self.fc_rgb(x)
        return torch.cat((rgb, alpha), dim=-1)
