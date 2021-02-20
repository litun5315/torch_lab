import torch
from MaxChannelPool import MaxChannelPool


def test_max_channel_pool(a):
    a_t = torch.FloatTensor(a)
    mcp = MaxChannelPool()
    a_t_maxp = mcp(a_t)
    print(a_t.shape)
    print(a_t)
    print('++++++++++++++++++++++++++++++++++')
    print(a_t_maxp.shape)
    print(a_t_maxp)


if __name__ == '__main__':
    a = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]]
    # a = torch.randn(8, 3, 255, 255)

    test_max_channel_pool(a)
