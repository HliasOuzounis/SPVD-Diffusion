from my_models.modules.sparse_utils import mixed_mix, PointTensor

import torch

def main():
    from torchsparse import SparseTensor

    a = SparseTensor(
        feats=torch.randn(10, 16),
        coords=torch.randint(0, 100, (10, 3)),
    )
    print(a, a.F)
    
    b = torch.randn(10, 16)
    b = b.to_sparse()
    print(b)
    

if __name__ == '__main__':
    main()

