from rl4co.envs.routing import ATSPEnv
from rl4co.models import MatNet
from rl4co.utils import RL4COTrainer

# Instantiate environment
env = ATSPEnv(generator_params=dict(num_loc=20))

# Create model
model = MatNet(
    env,
    baseline="shared",
    train_data_size=10,
    val_data_size=10,
    test_data_size=10,
)

# Instantiate Trainer and fit
trainer = RL4COTrainer(max_epochs=1, devices=1, accelerator="gpu")
trainer.fit(model)
trainer.test(model)