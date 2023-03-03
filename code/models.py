import torch


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