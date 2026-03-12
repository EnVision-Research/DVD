import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        training_target='x',
        training_weight_type='default'
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.training_weight_type = training_weight_type

        # Initialize basic attributes
        self.target = None
        self.timesteps = None
        self.sigmas = None
        self.linear_timesteps_weights = None
        self.training = False

        self.set_training_target(training_target=training_target)
        self.set_training_weight(training_weight_type=training_weight_type)

    def set_training_target(self, training_target='x'):
        self.target = training_target

    def set_training_weight(self, training_weight_type):
        valid_types = ["default", "equal", "early", "late"]
        assert training_weight_type in valid_types, \
            f"training_weight_type must be one of {valid_types}"
        self.training_weight_type = training_weight_type

    def set_timesteps(
        self,
        num_inference_steps=100,  # Kept for signature compatibility if needed
        denoising_strength=1.0,   # Kept for signature compatibility if needed
        training=False,
        shift=None,
        denoise_step=0.5,
        **kwargs
    ):
        if shift is not None:
            self.shift = shift

        self.training = training

        # As requested: single value calculations
        # timestep = 1000 * denoise_step
        # sigma = timestep / 1000  (which simplifies to just denoise_step)
        # weight = 1.0

        ts_val = self.num_train_timesteps * denoise_step
        sigma_val = ts_val / self.num_train_timesteps
        weight_val = 1.795

        # Create tensors with a single value
        self.timesteps = torch.tensor([ts_val], dtype=torch.float32)
        self.sigmas = torch.tensor([sigma_val], dtype=torch.float32)

        if self.training:
            self.linear_timesteps_weights = torch.tensor(
                [weight_val], dtype=torch.float32)
        else:
            self.linear_timesteps_weights = None

    def step(self, model_output, sample, to_final=False, **kwargs):
        if self.target == 'x':
            return model_output
        elif self.target == 'flow':
            return sample - model_output

    def training_target(self, sample, noise, timestep):
        if self.target == 'x':
            return sample
        elif self.target == 'flow':
            target = noise - sample
            return target

    def training_weight(self, timestep):
        # Since linear_timesteps_weights only has one value now,
        # we can just return it.
        # (Assuming the logic intends to fetch the unified weight)
        if self.linear_timesteps_weights is not None:
            return self.linear_timesteps_weights[0]
        return 1.0


if __name__ == "__main__":
    scheduler = FlowMatchScheduler()
    scheduler.set_training_weight("default")
    scheduler.set_timesteps(
        num_inference_steps=1,
        training=True,
        schedule_mode="default",
        denoise_step=1,
        shift=5
    )

    for step, sigma, weight in zip(scheduler.timesteps, scheduler.sigmas, scheduler.linear_timesteps_weights):
        print(
            f"Step: {step.item()}, Sigma: {sigma.item()}, Weight: {weight.item()}")
