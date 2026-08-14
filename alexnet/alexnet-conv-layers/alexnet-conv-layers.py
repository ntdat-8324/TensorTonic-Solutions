import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    # YOUR CODE HERE
    #image (2, 224, 224, 3)
    batch_size, height, width, in_channels = image.shape

    kernel_size = 11
    stride = 4
    padding = 2
    out_channels = 96

    image = np.pad(
        image,
        pad_width=(
                (0, 0),             
                (padding, padding), 
                (padding, padding),  
                (0, 0)               
            ),
            mode="constant"
        )
    weights = np.random.randn(kernel_size, kernel_size, in_channels, out_channels)

    bias = np.zeros(out_channels)

    out_height = (height - kernel_size + 2*padding) // stride + 1
    out_width = (width - kernel_size + 2*padding) // stride + 1

    output = np.zeros((batch_size, out_height, out_width, out_channels))

    for b in range(batch_size):
        for i in range(out_height):
            for j in range(out_width):

                h_start = i*stride
                w_start = j*stride

                patch = image[b, h_start:(h_start+kernel_size), w_start:(w_start+kernel_size), :]

                output[b, i, j, :] = (
                    np.tensordot(
                        patch,
                        weights,
                        axes=((0, 1, 2), (0, 1, 2))
                    )
                    + bias
                )
    
    return output