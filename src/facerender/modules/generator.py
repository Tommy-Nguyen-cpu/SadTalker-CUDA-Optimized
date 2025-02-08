import torch
from torch import nn
import torch.nn.functional as F
from src.facerender.modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from src.facerender.modules.dense_motion import DenseMotionNetwork


class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(out_features, kernel_size=3, padding=1))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            # output_dict["deformed"] = self.deform_input(source_image, deformation)  # 3d deformation cannot deform 2d image

        # Decoding part
        out = self.resblocks_2d(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        
    def forward(self, feature):
        seg = feature
        x = self.fc(feature)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        x = self.G_middle_2(x, seg)
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        x = self.up(x)                
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)                
        x = self.up_1(x, seg)         # 64, 256, 256

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        # x = torch.tanh(x)
        x = F.sigmoid(x)
        
        return x


class OcclusionAwareSPADEGenerator(nn.Module):

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.decoder = SPADEDecoder()

    def grid_sample_5d(self, input, grid, mode='linear', padding_mode='zeros', align_corners=True):
        """
        Custom implementation of 5D grid sampling.

        Args:
            input (torch.Tensor): Input tensor of shape (N, C, D, H, W, T).
            grid (torch.Tensor): Sampling grid of shape (N, D_out, H_out, W_out, T_out, 5).
            mode (str): Interpolation mode ('linear' for multilinear).
            padding_mode (str): Padding mode ('zeros', 'border', or 'reflection').
            align_corners (bool): Align corners flag for coordinate normalization.

        Returns:
            torch.Tensor: Output tensor of shape (N, C, D_out, H_out, W_out, T_out).
        """
        C, D, H, W, T = input.shape
        D_out, H_out, W_out, T_out, _ = grid.shape

        # Normalize grid to match the input dimensions
        def normalize(coords, size):
            if align_corners:
                return (coords + 1) * (size - 1) / 2
            else:
                return ((coords + 1) * size - 1) / 2

        # Normalize grid coordinates
        grid = torch.stack([normalize(grid[..., i], size)
                            for i, size in enumerate([T, W, H, D, C])], dim=-1)

        # Extract grid coordinates
        t, w, h, d, c = [grid[..., i] for i in range(5)]

        # Clamp coordinates within bounds
        t0, t1 = t.floor().long(), t.ceil().long()
        w0, w1 = w.floor().long(), w.ceil().long()
        h0, h1 = h.floor().long(), h.ceil().long()
        d0, d1 = d.floor().long(), d.ceil().long()
        c0, c1 = c.floor().long(), c.ceil().long()

        # Ensure coordinates stay within valid range
        def safe_index(x, max_size):
            return torch.clamp(x, 0, max_size - 1)

        t0, t1 = safe_index(t0, T), safe_index(t1, T)
        w0, w1 = safe_index(w0, W), safe_index(w1, W)
        h0, h1 = safe_index(h0, H), safe_index(h1, H)
        d0, d1 = safe_index(d0, D), safe_index(d1, D)

        # Perform multilinear interpolation
        def interpolate(input, t, w, h, d):
            return input[:, :, d, h, w, t]

        c00 = interpolate(input, t0, w0, h0, d0)
        c01 = interpolate(input, t0, w0, h1, d0)
        c10 = interpolate(input, t0, w1, h0, d0)
        c11 = interpolate(input, t0, w1, h1, d0)
        c0 = (1 - w) * c00 + w * c10
        c1 = (1 - w) * c01 + w * c11
        c = (1 - h) * c0 + h * c1

        # Final output
        output = c.unsqueeze(1)
        return output

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        b, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)

        print(f"Output shape: input:{inp.shape}, grid:{deformation.shape}")
        # inp: torch.Size([2, 32, 16, 64, 64]), grid: torch.Size([2, 16, 64, 64, 3])
        # The inner most tensor of the grid (represented by 3) tells us how to fill in missing values (i.e., index or interpolate) into the input tensor.
        # In other words: grid[2, 16, 64, 64] tells us how to interpolate into inp[2,:,16,64,64]

        # out = Initialized to be copy of inp.
        out = inp.detach().clone()
        indices = inp.to_sparse().indices().to(torch.float16)
        print(f"indices: {torch.max(indices)}")
        mean = torch.mean(indices)
        std = torch.std(indices)
        print(f"mean: {mean}, std: {std}")
        # Iterate through batch b of size 2
        for b_idx in range(b):
            # Iterate over depth d of size 16
            for d_idx in range(d):
                # Iterate over height h of size 64
                for h_idx in range(h):
                    # Iterate over width w of size 64
                    for w_idx in range(w):
                        # (x,y,z) coordinate tensors <- retrieved from grid[b, d, h, w]
                        # TODO: Used inp indices tensor as a way of reversing normalized deformation tensor.
                        # Mean and standard deviation calculated using inp indices, but now its saying its out of bounds. Will have to investigate.
                        coordinate = deformation[b_idx, d_idx, h_idx, w_idx]
                        x = abs(coordinate[0] * std) + mean
                        y = abs(coordinate[1] * std) + mean
                        z = abs(coordinate[2] * std) + mean
                        print(f"deformation: {deformation[b_idx, d_idx, h_idx, w_idx]}")
                        print(f"{x}, {y}, {z}")
                        # print(f"coordinates: {float(coordinate[0])}, {float(coordinate[1])}, {(float(coordinate[2]))}")
                        # (x,y,z) unnormalized <- unnormalized using inp size (2, 32, 16, 64, 64) ? Doc says it should be normalized but reddit says otherwise. Doc is probably more trustworthy.
                        out[b_idx, :, d_idx, h_idx, w_idx] = F.interpolate(inp, size=(int(x),int(y), int(z)))
        
        # TODO: It looks like 4D STILL doesn't work in TensorRT (says input is not equals to 4D even though it is?).
        # Will have to fall back on implementing grid_sample from scratch.
        return F.grid_sample(inp, deformation) # self.grid_sample_5d(inp, deformation)

    # TODO: It's very obvious that the bulk of the time overhead is the generator. Will focus on making it more efficient.
    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            # import pdb; pdb.set_trace()

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            # occlusion_map = torch.where(occlusion_map < 0.95, 0, occlusion_map)
            
            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out
        
        return output_dict