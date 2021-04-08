import torch


class MaxChannelPool(torch.nn.Module):
    def __init__(self, kernel_size=7, stride=2, padding=3, dilation=1,
                 return_indices=False, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        self.compression = 2
        self.output = None

    def forward(self, input):
        n, c, w, h = input.size()
        # Add padding to input so work with kernal size
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, self.padding, self.padding), "constant", 0)
        output = torch.stack([
            torch.stack(
                [torch.max(input[x][index:index + self.kernel_size - 1], axis=0)[0]  # Get max from kernal size
                 for index in range(0, input.size()[1] - self.kernel_size, self.stride)])  # Move stride
            for x in range(n)])  # Do work for each image in batch

        #     return output.cuda()
        return output
