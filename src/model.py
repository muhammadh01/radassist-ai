from monai.networks.nets import UNet

def get_model():
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

if __name__ == "__main__":
    import torch
    model = get_model()
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
