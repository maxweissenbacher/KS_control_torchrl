import torch
from icecream import ic
from utils import make_ks_env, make_tqc_agent
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

    print('stop here')

    train_env, eval_env = make_ks_env(cfg)

    # Doing a simple rollout
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

    # Checking if memory resets to zero correctly
    td = eval_env.reset()
    print(f"Memory before applying actor:")
    print(td["memory"])
    init_memory = td["memory"]
    actor(td)
    print(f"Memory after applying actor:")
    print(td["memory"])
    print(f"Difference between memories")
    print((td["memory"]-init_memory))

    model, _ = make_tqc_agent(cfg, train_env, eval_env)
    steps = 100
    rollout = eval_env.rollout(max_steps=steps, policy=model[0])
    for i in range(steps):
        print(rollout["next", "memory"][i].mean().item())

    print('stop here')


if __name__ == '__main__':
    main()
