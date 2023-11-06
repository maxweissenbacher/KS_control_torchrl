import torch
from icecream import ic
from utils import make_ks_env
from models.attention.self_attention import MultiHeadAttention, SelfAttentionMemoryActor
import hydra


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg):
    n_heads = 5
    size_memory = 30
    num_memories = 20
    batchsize = [100, 100]

    # M is the memory
    # M ... batch_size x num_memories x size_memory
    M = torch.zeros([*batchsize, num_memories, size_memory])
    x = torch.ones([*batchsize, size_memory])

    MultiHeadAttention(size_memory, n_heads, 'cpu')(M, x)

    train_env, eval_env = make_ks_env(cfg)

    # Doing a simple rollout with zero actions
    td = eval_env.reset()
    rollout = eval_env.rollout(max_steps=3)

    ic(rollout)

    actor = SelfAttentionMemoryActor(
        cfg,
        action_spec=eval_env.action_spec,
        out_key="actor_net_out",
    )

    actor(rollout)

    ic(actor(rollout))


if __name__ == '__main__':
    main()
