import numpy as np

def correct_val_steps(train_steps, val_steps, val_freq):
    """
    Adjusts validation steps to be continuous across epochs.
    
    Args:
    train_steps (int): Total number of training steps.
    val_steps (np.array): Array of original validation time steps.
    val_freq (int): Number of training steps between validation steps.
    
    Returns:
    np.array: Corrected validation time steps.
    """
    corrected_steps = np.copy(val_steps)
    current_epoch_start = val_steps[0]

    for i in range(1, len(val_steps)):
        if val_steps[i] <= val_steps[i - 1]:  # Detect reset indicating new epoch
            current_epoch_start = corrected_steps[i - 1] + val_freq
        corrected_steps[i] = current_epoch_start + (val_steps[i] - val_steps[0])

    return corrected_steps

val_steps_path = '../transformer/plots_halfcheetah_obs_2l_80bsz/transformer_val_time_steps.npy'
train_steps_total = 93  # Total number of training steps
val_steps = np.load(val_steps_path)
val_freq = 30  # Validation frequency per epoch

corrected_val_steps = correct_val_steps(train_steps_total, val_steps, val_freq)

# Save and print location
rescaled_val_steps_path = '../transformer/plots_halfcheetah_obs_2l_80bsz/transformer_val_corrected_time_steps.npy'
np.save(rescaled_val_steps_path, corrected_val_steps)
print(f"Rescaled validation time steps saved to {rescaled_val_steps_path}")
