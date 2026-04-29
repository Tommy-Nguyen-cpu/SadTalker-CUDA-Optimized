import torch

def trilinear_sampler(
    input_vol,
    grid,
    align_corners=False,
    padding_mode="zeros",
):
    """
    input_vol: (N, C, D, H, W)
    grid:      (N, D_out, H_out, W_out, 3), coords in [-1, 1]
    """

    N, C, D, H, W = input_vol.shape
    _, D_out, H_out, W_out, _ = grid.shape

    # --- coordinate transform ---
    if align_corners:
        x = (grid[..., 0] + 1) * (W - 1) / 2
        y = (grid[..., 1] + 1) * (H - 1) / 2
        z = (grid[..., 2] + 1) * (D - 1) / 2
    else:
        x = ((grid[..., 0] + 1) * W - 1) / 2
        y = ((grid[..., 1] + 1) * H - 1) / 2
        z = ((grid[..., 2] + 1) * D - 1) / 2

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    xd = x - x0
    yd = y - y0
    zd = z - z0

    # flatten for gather
    flat = input_vol.reshape(N, C, D * H * W)

    def gather(ix, iy, iz):
        valid = (
            (ix >= 0) & (ix <= W - 1) &
            (iy >= 0) & (iy <= H - 1) &
            (iz >= 0) & (iz <= D - 1)
        )

        ix = ix.clamp(0, W - 1)
        iy = iy.clamp(0, H - 1)
        iz = iz.clamp(0, D - 1)

        lin = (iz * H * W + iy * W + ix).long()
        lin = lin.view(N, -1)

        gathered = torch.gather(
            flat,
            2,
            lin.unsqueeze(1).expand(-1, C, -1)
        ).view(N, C, D_out, H_out, W_out)

        if padding_mode == "zeros":
            gathered = gathered * valid.unsqueeze(1)

        return gathered

    # 8 corners
    c000 = gather(x0, y0, z0)
    c001 = gather(x0, y0, z1)
    c010 = gather(x0, y1, z0)
    c011 = gather(x0, y1, z1)
    c100 = gather(x1, y0, z0)
    c101 = gather(x1, y0, z1)
    c110 = gather(x1, y1, z0)
    c111 = gather(x1, y1, z1)

    # weights
    w000 = (1 - xd) * (1 - yd) * (1 - zd)
    w001 = (1 - xd) * (1 - yd) * zd
    w010 = (1 - xd) * yd * (1 - zd)
    w011 = (1 - xd) * yd * zd
    w100 = xd * (1 - yd) * (1 - zd)
    w101 = xd * (1 - yd) * zd
    w110 = xd * yd * (1 - zd)
    w111 = xd * yd * zd

    # expand
    w000 = w000.unsqueeze(1)
    w001 = w001.unsqueeze(1)
    w010 = w010.unsqueeze(1)
    w011 = w011.unsqueeze(1)
    w100 = w100.unsqueeze(1)
    w101 = w101.unsqueeze(1)
    w110 = w110.unsqueeze(1)
    w111 = w111.unsqueeze(1)

    out = (
        w000 * c000 + w001 * c001 +
        w010 * c010 + w011 * c011 +
        w100 * c100 + w101 * c101 +
        w110 * c110 + w111 * c111
    )

    return out.to(input_vol.dtype)