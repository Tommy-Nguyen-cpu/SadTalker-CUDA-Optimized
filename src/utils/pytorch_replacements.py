import torch

def trilinear_sampler(input_vol, grid):
        """
        input_vol: (N, C, D, H, W)
        grid:      (N, D_out, H_out, W_out, 3), coords in [-1,1]
        returns:   (N, C, D_out, H_out, W_out)
        """
        N, C, D, H, W = input_vol.shape
        _, D_out, H_out, W_out, _ = grid.shape

        # normalize coords
        x = (grid[..., 0] + 1) * 0.5 * (W - 1)
        y = (grid[..., 1] + 1) * 0.5 * (H - 1)
        z = (grid[..., 2] + 1) * 0.5 * (D - 1)

        # corner indices
        x0 = torch.floor(x).long().clamp(0, W-1);   x1 = (x0 + 1).clamp(0, W-1)
        y0 = torch.floor(y).long().clamp(0, H-1);   y1 = (y0 + 1).clamp(0, H-1)
        z0 = torch.floor(z).long().clamp(0, D-1);   z1 = (z0 + 1).clamp(0, D-1)

        # gather helper
        def g(ix, iy, iz):
            # ix, iy, iz each are (N, D_out, H_out, W_out)
            b = torch.arange(N, device=input_vol.device)[:, None, None, None]
            b = b.expand(-1, D_out, H_out, W_out)
            # this returns (N, D_out, H_out, W_out, C):
            tmp = input_vol[b, :, iz, iy, ix]       # → (N, D_out, H_out, W_out, C)
            tmp = tmp.permute(0, 4, 1, 2, 3)         # → (N, C, D_out, H_out, W_out)
            # either of these will now work:
            return tmp.contiguous().view(N, C, D_out, H_out, W_out)

        # gather corner voxels
        c000 = g(x0, y0, z0)
        c001 = g(x0, y0, z1)
        c010 = g(x0, y1, z0)
        c011 = g(x0, y1, z1)
        c100 = g(x1, y0, z0)
        c101 = g(x1, y0, z1)
        c110 = g(x1, y1, z0)
        c111 = g(x1, y1, z1)

        # compute deltas
        x0f, y0f, z0f = x0.float(), y0.float(), z0.float()
        xd, yd, zd = x - x0f, y - y0f, z - z0f

        # raw weights
        w000 = (1 - xd) * (1 - yd) * (1 - zd)
        w001 = (1 - xd) * (1 - yd) * zd
        w010 = (1 - xd) * yd       * (1 - zd)
        w011 = (1 - xd) * yd       * zd
        w100 = xd       * (1 - yd) * (1 - zd)
        w101 = xd       * (1 - yd) * zd
        w110 = xd       * yd       * (1 - zd)
        w111 = xd       * yd       * zd

        # unsqueeze for channel broadcast
        w000 = w000.unsqueeze(1)
        w001 = w001.unsqueeze(1)
        w010 = w010.unsqueeze(1)
        w011 = w011.unsqueeze(1)
        w100 = w100.unsqueeze(1)
        w101 = w101.unsqueeze(1)
        w110 = w110.unsqueeze(1)
        w111 = w111.unsqueeze(1)

        # combine corners with weights
        out = (
            w000 * c000 + w001 * c001 +
            w010 * c010 + w011 * c011 +
            w100 * c100 + w101 * c101 +
            w110 * c110 + w111 * c111
        )
        
        return out.to(input_vol.dtype)