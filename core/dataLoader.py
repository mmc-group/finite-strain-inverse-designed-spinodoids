from .__importList__ import *
from .config import *

# Theta string splitting function
def split_theta_strings(theta_string):
    return [int(theta_string[i:i+2]) for i in range(0, 6, 2)]

def load_data():
    n_output = 3
    os.makedirs('normalization/', exist_ok=True)
    directions = ['X', 'Y', 'Z']

    # Precompute number of files and dataset size
    files_in_dir = os.listdir(join(data_path, directions[0]))
    num_files = len([f for f in files_in_dir if isfile(join(data_path, directions[0], f))])
    dataset_size = num_files * time_steps

    # Initialize datasets
    theta_dataset = torch.zeros((dataset_size, 3))
    sigma_dataset = torch.zeros((dataset_size, 3))
    eps_dataset = torch.zeros((dataset_size, 1))

    for j, direction in enumerate(directions):
        data_path_tmp = join(data_path, direction)
        files = [f for f in os.listdir(data_path_tmp) if isfile(join(data_path_tmp, f))]

        for i, f in enumerate(files):
            # Read stress and epsilon data
            tes = pd.read_csv(join(data_path_tmp, f), header=None)
            sigma = tes[1].to_numpy()
            eps = tes[0].to_numpy()

            # Truncate up to unload index
            unload_idx = np.argmax(eps)
            eps = eps[:unload_idx].astype(float)
            sigma = sigma[:unload_idx].astype(float)

            # Interpolate stress data
            func = interpolate.interp1d(eps, sigma)
            eps_new = np.linspace(1e-3, max_eps, time_steps)
            sigma_new = func(eps_new)

            # Get thetas and convert to tensor
            thetas_int = split_theta_strings(f[:6])

            # Update datasets
            start_idx = i * time_steps
            end_idx = start_idx + time_steps

            sigma_dataset[start_idx:end_idx, j] = torch.from_numpy(sigma_new).float()
            eps_dataset[start_idx:end_idx, 0] = torch.from_numpy(eps_new).float()
            theta_dataset[start_idx:end_idx, :] = torch.tensor(thetas_int) / 70.

    # Normalize datasets
    sigma_scaling = Normalization(sigma_dataset.view(-1, 1))
    eps_scaling = Normalization(eps_dataset.view(-1, 1))

    with open('normalization/sigma_scaling.pickle', 'wb') as file_:
        pickle.dump(sigma_scaling, file_, -1)
    with open('normalization/eps_scaling.pickle', 'wb') as file_:
        pickle.dump(eps_scaling, file_, -1)

    sigma_dataset = sigma_scaling.normalize(sigma_dataset.view(-1, 1)).view(-1, n_output)
    eps_dataset = eps_scaling.normalize(eps_dataset)

    idx_lam = 8
    idx_col = 38
    idx_cub = 93


    # Define training and validation datasets
    train_indices = torch.cat([
        torch.arange(0, idx_lam * time_steps),
        torch.arange((idx_lam+1) * time_steps, idx_col * time_steps),
        torch.arange((idx_col+1) * time_steps, idx_cub * time_steps),
        torch.arange((idx_cub+1) * time_steps, dataset_size)
    ])
    val_indices = torch.cat([
        torch.arange(idx_lam * time_steps, (idx_lam+1) * time_steps),
        torch.arange(idx_col * time_steps, (idx_col+1) * time_steps),
        torch.arange(idx_cub * time_steps, (idx_cub+1) * time_steps)
    ])

    train_set = TensorDataset(
        theta_dataset[train_indices].float(),
        eps_dataset[train_indices].float(),
        sigma_dataset[train_indices].float()
    )
    val_set = TensorDataset(
        theta_dataset[val_indices].float(),
        eps_dataset[val_indices].float(),
        sigma_dataset[val_indices].float()
    )

    return train_set, val_set, sigma_scaling