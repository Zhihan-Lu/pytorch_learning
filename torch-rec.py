import torch

import torchrec
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

ebc = torchrec.EmbeddingBagCollection(
    device="cpu",
    tables=[
        torchrec.EmbeddingBagConfig(
            name="product_table",
            embedding_dim=16,
            num_embeddings=4096,
            feature_names=["product"],
            pooling=torchrec.PoolingType.SUM,
        ),
        torchrec.EmbeddingBagConfig(
            name="user_table",
            embedding_dim=16,
            num_embeddings=4096,
            feature_names=["user"],
            pooling=torchrec.PoolingType.SUM,
        )
    ]
)

product_jt = JaggedTensor(
    values=torch.tensor([1, 2, 1, 5]), lengths=torch.tensor([3, 1])
)
user_jt = JaggedTensor(values=torch.tensor([2, 3, 4, 1]), lengths=torch.tensor([2, 2]))

# Q1: How many batches are there, and which values are in the first batch for product_jt and user_jt?
kjt = KeyedJaggedTensor.from_jt_dict({"product": product_jt, "user": user_jt})

print("Call EmbeddingBagCollection Forward: ", ebc(kjt))