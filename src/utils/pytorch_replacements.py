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
    returns:   (N, C, D_out, H_out, W_out)

    This is a manual trilinear sampler intended to match torch.nn.functional.grid_sample
    as closely as possible for 5D input.
    """

    if padding_mode not in ("zeros", "border"):
        raise ValueError("padding_mode must be 'zeros' or 'border'")

    N, C, D, H, W = input_vol.shape
    _, D_out, H_out, W_out, _ = grid.shape

    # Match grid_sample coordinate transform
    if align_corners:
        x = (grid[..., 0] + 1) * (W - 1) / 2
        y = (grid[..., 1] + 1) * (H - 1) / 2
        z = (grid[..., 2] + 1) * (D - 1) / 2
    else:
        x = ((grid[..., 0] + 1) * W - 1) / 2
        y = ((grid[..., 1] + 1) * H - 1) / 2
        z = ((grid[..., 2] + 1) * D - 1) / 2

    # Base corners
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    z0 = torch.floor(z)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    # Interpolation deltas
    xd = x - x0
    yd = y - y0
    zd = z - z0

    # Flatten input for easier gather: (N, C, D*H*W)
    flat = input_vol.reshape(N, C, D * H * W)

    b = torch.arange(N, device=input_vol.device)[:, None, None, None]
    b = b.expand(N, D_out, H_out, W_out)

    def gather(ix, iy, iz):
        """
        ix, iy, iz: (N, D_out, H_out, W_out)
        Returns:    (N, C, D_out, H_out, W_out)
        """
        if padding_mode == "zeros":
            valid = (
                (ix >= 0) & (ix <= W - 1) &
                (iy >= 0) & (iy <= H - 1) &
                (iz >= 0) & (iz <= D - 1)
            )
        else:
            valid = None

        ix_safe = ix.clamp(0, W - 1)
        iy_safe = iy.clamp(0, H - 1)
        iz_safe = iz.clamp(0, D - 1)

        lin = (iz_safe * H * W + iy_safe * W + ix_safe).long()
        lin = lin.reshape(N, -1)  # (N, D_out*H_out*W_out)

        gathered = torch.gather(
            flat,
            2,
            lin.unsqueeze(1).expand(-1, C, -1)
        )
        gathered = gathered.reshape(N, C, D_out, H_out, W_out)

        if valid is not None:
            gathered = gathered * valid.unsqueeze(1).to(gathered.dtype)

        return gathered

    # Corner samples
    c000 = gather(x0.long(), y0.long(), z0.long())
    c001 = gather(x0.long(), y0.long(), z1.long())
    c010 = gather(x0.long(), y1.long(), z0.long())
    c011 = gather(x0.long(), y1.long(), z1.long())
    c100 = gather(x1.long(), y0.long(), z0.long())
    c101 = gather(x1.long(), y0.long(), z1.long())
    c110 = gather(x1.long(), y1.long(), z0.long())
    c111 = gather(x1.long(), y1.long(), z1.long())

    # Weights
    w000 = (1 - xd) * (1 - yd) * (1 - zd)
    w001 = (1 - xd) * (1 - yd) * zd
    w010 = (1 - xd) * yd * (1 - zd)
    w011 = (1 - xd) * yd * zd
    w100 = xd * (1 - yd) * (1 - zd)
    w101 = xd * (1 - yd) * zd
    w110 = xd * yd * (1 - zd)
    w111 = xd * yd * zd

    # Broadcast over channel
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