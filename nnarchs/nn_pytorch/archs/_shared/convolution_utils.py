from torch import nn




def is_depthwise_3x3(m: nn.Module) -> bool:
    """
    A plain nn.Conv2d (not a subclass, whose forward may differ) computing a
    depthwise 3x3 convolution with stride 1 and zero padding 1.
    """
    return (
        type(m) is nn.Conv2d
        and m.groups == m.in_channels == m.out_channels
        and m.kernel_size == (3, 3)
        and m.stride == (1, 1)
        and m.padding == (1, 1)
        and m.dilation == (1, 1)
        and m.padding_mode == "zeros"
    )
