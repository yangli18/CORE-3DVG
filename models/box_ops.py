import torch


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    return torch.stack(
        [
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            z_c - 0.5 * d,
            x_c + 0.5 * w,
            y_c + 0.5 * h,
            z_c + 0.5 * d,
        ],
        dim=-1,
    )


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    x_a = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    y_a = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    z_a = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    x_b = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    y_b = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    z_b = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(x_b - x_a, 0)
        * torch.clamp(y_b - y_a, 0)
        * torch.clamp(z_b - z_a, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def _batch_iou3d_par(box_a, box_b):
    def _batch_intersect_par(a, b):
        x_a = torch.max(a[..., [0]], b[:, None, :, 0])
        y_a = torch.max(a[..., [1]], b[:, None, :, 1])
        z_a = torch.max(a[..., [2]], b[:, None, :, 2])
        x_b = torch.min(a[..., [3]], b[:, None, :, 3])
        y_b = torch.min(a[..., [4]], b[:, None, :, 4])
        z_b = torch.min(a[..., [5]], b[:, None, :, 5])
        return (
            torch.clamp(x_b - x_a, 0)
            * torch.clamp(y_b - y_a, 0)
            * torch.clamp(z_b - z_a, 0)
        )

    def _batch_volume_par(box):
        return (
            (box[..., 3] - box[..., 0])
            * (box[..., 4] - box[..., 1])
            * (box[..., 5] - box[..., 2])
        )

    intersection = _batch_intersect_par(box_a, box_b)
    vol_a = _batch_volume_par(box_a)
    vol_b = _batch_volume_par(box_b)
    union = vol_a[..., None] + vol_b[:, None, :] - intersection
    return intersection / union, union
