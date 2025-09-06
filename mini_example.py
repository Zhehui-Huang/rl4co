import torch

from rl4co.envs.routing import ATSPEnv
from rl4co.models import MatNet
from rl4co.utils.trainer import RL4COTrainer

# Instantiate environment
env = ATSPEnv(generator_params=dict(num_loc=20))

# Create model
model = MatNet(
    env,
    baseline="shared",
    train_data_size=100000,
    val_data_size=100000,
    test_data_size=100000,
)

# Greedy rollouts over untrained policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[3]).to(device)
policy = model.policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_untrained = out['actions'].cpu().detach()
rewards_untrained = out['reward'].cpu().detach()

for i in range(3):
    print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.3f}")
    env.render(td_init[i], actions_untrained[i])

trainer = RL4COTrainer(
    max_epochs=10,
    accelerator="gpu",
    devices=1,
    logger=None,
)

trainer.fit(model)

# Greedy rollouts over trained model (same states as previous plot)
policy = model.policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_trained = out['actions'].cpu().detach()

# Plotting
import matplotlib.pyplot as plt
for i, td in enumerate(td_init):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    env.render(td, actions_untrained[i], ax=axs[0])
    env.render(td, actions_trained[i], ax=axs[1])
    axs[0].set_title(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
    axs[1].set_title(r"Trained $\pi_\theta$" + f" | Cost = {-out['reward'][i].item():.3f}")
    plt.tight_layout()
    plt.show()