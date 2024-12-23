import sys
sys.path.insert(0, "../")  # Add the parent directory to the system path
from core.dataLoader import *  # Import data loading utilities
from model import *            # Import model definitions
from tqdm import tqdm          # Import progress bar for loops
from core.config import *      # Import configuration settings
from core.utils import *       # Import utility functions

# Parse command-line arguments for direction and improve_checkpoint
try:
    direction = sys.argv[2]
    if sys.argv[1] == 'train':
        improve_checkpoint = eval(sys.argv[3])
except:
    print('Insufficient arguments. Try: main.py (mode=)[train | test] (direction=)[X | Y | Z] (improve_checkpoint)[True | False]')
    sys.exit(0)

if __name__ == '__main__':
    torch.cuda.empty_cache()  # Clear GPU memory cache
    loss_best = 1e9           # Initialize best loss with a very high value

    # Create a directory to save models if it doesn't exist
    pathlib.Path('models').mkdir(exist_ok=True)

    # Load and preprocess the dataset
    print('Loading dataset...')
    train_set, val_set, sigma_scaler = load_data()
    if sys.argv[1] == 'opt':
        train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=time_steps, shuffle=False)
    else:
        train_data_loader = DataLoader(dataset=train_set, num_workers=numWorkers, batch_size=batchSize, shuffle=True)
    test_data_loader = DataLoader(dataset=val_set, num_workers=numWorkers, batch_size=time_steps)
    print('Finished loading dataset.')
    print('\n-------------------------------------')

    # Initialize PICNN models for different inputs and assign them to the device
    lam1 = PICNN(n_input=1+dim_y, n_hidden=n_hidden_lam, n_output=n_output).to(device)
    col1 = PICNN(n_input=2+dim_y, n_hidden=n_hidden_col, n_output=n_output).to(device)
    cub1 = PICNN(n_input=3+dim_y, n_hidden=n_hidden_cub, n_output=n_output).to(device)
    lam2 = PICNN(n_input=1+dim_y, n_hidden=n_hidden_lam, n_output=n_output).to(device)
    col2 = PICNN(n_input=2+dim_y, n_hidden=n_hidden_col, n_output=n_output).to(device)
    cub2 = PICNN(n_input=3+dim_y, n_hidden=n_hidden_cub, n_output=n_output).to(device)
    b_net = FFNN(n_input=3, n_hidden=bv_hid, n_output=n_output).to(device)
    v_net = FFNN(n_input=3, n_hidden=bv_hid, n_output=n_output).to(device)
    kT_net = FFNN(n_input=3, n_hidden=bv_hid, n_output=n_output).to(device)

    if sys.argv[1] == 'train':
        # Initialize weights for FFNN models
        v_net.apply(init_weights)
        b_net.apply(init_weights)
        kT_net.apply(init_weights)

        # Define the optimizer for all model parameters
        optimizer = torch.optim.Adam(
            list(lam1.parameters()) + list(col1.parameters()) + list(cub1.parameters()) +
            list(lam2.parameters()) + list(col2.parameters()) + list(cub2.parameters()) +
            list(b_net.parameters()) + list(v_net.parameters()) + list(kT_net.parameters()),
            lr=lr
        )

        history = []

        if direction == 'X' or improve_checkpoint:
            if direction == 'X':
                try:
                    checkpoint = torch.load('models/Y.pt', weights_only=False)
                except:
                    print('Train model in Y-direction first. Model in X will be using weights of model in Y as starting point.')
                    sys.exit(0)
                    
            else:
                try:
                    checkpoint = torch.load('models/'+direction+'.pt', weights_only=False)
                except:
                    print('Model in '+direction+' does not exist yet.')
                    sys.exit(0)
                    
            lam1.load_state_dict(checkpoint['lam1'])
            col1.load_state_dict(checkpoint['col1'])
            cub1.load_state_dict(checkpoint['cub1'])
            lam2.load_state_dict(checkpoint['lam2'])
            col2.load_state_dict(checkpoint['col2'])
            cub2.load_state_dict(checkpoint['cub2'])
            b_net.load_state_dict(checkpoint['b_net'])
            v_net.load_state_dict(checkpoint['v_net'])
            kT_net.load_state_dict(checkpoint['kT_net'])

        # Set models to training mode
        lam1.train().to(device)
        col1.train().to(device)
        cub1.train().to(device)
        lam2.train().to(device)
        col2.train().to(device)
        cub2.train().to(device)
        b_net.train().to(device)
        v_net.train().to(device)

        # Initialize training and testing history lists
        train_history = []
        test_history = []
        epoch_list = []

        # Training loop
        for epoch_iter in range(epochs):
            epoch_list.append(epoch_iter)
            train_loss = 0
            test_loss = 0

            # Training step
            for batch in tqdm(train_data_loader):
                lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, optimizer, train_loss_batch = predict(
                    batch, lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, optimizer, device, mode='train', direction=direction
                )
                train_loss += train_loss_batch / len(train_data_loader)

            # Testing step
            for batch in test_data_loader:
                test_loss_batch = test(
                    batch, lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, direction, sigma_scaler, device, plot=True
                )
                test_loss += test_loss_batch / len(test_data_loader)

            # Print epoch accuracy
            print("| {}:{}/{} | EpochTrainMAPE: {:.2%} | EpochTestMAPE: {:.2%}".format("Epoch", epoch_iter, epochs, train_loss, test_loss))
            train_history.append(train_loss)
            test_history.append(test_loss)

            # Save the best model based on training loss
            if test_loss < loss_best:
                loss_best = test_loss
                torch.save({
                    'lam1': lam1.state_dict(),
                    'col1': col1.state_dict(),
                    'cub1': cub1.state_dict(),
                    'lam2': lam2.state_dict(),
                    'col2': col2.state_dict(),
                    'cub2': cub2.state_dict(),
                    'b_net': b_net.state_dict(),
                    'v_net': v_net.state_dict(),
                    'kT_net': kT_net.state_dict()
                }, 'models/' + direction + '.pt')

            # Plot and save the training and testing loss curves
            plt.plot(epoch_list, train_history, label='Train')
            plt.plot(epoch_list, test_history, label='Test')
            plt.legend()
            plt.savefig('figures/loss-histories/loss_history_' + direction + '.png', dpi=350)
            plt.close()

        print('\n-------------------------------------')
        torch.cuda.empty_cache()  # Clear GPU memory cache
        print('Finished.')

    elif sys.argv[1] == 'test':
        # Data loaders for testing
        train_data_loader = DataLoader(dataset=train_set, batch_size=time_steps, shuffle=False)
        test_data_loader = DataLoader(dataset=val_set, batch_size=time_steps, shuffle=False)

        # Load model checkpoints
        try:
            checkpoint = torch.load('models/' + direction + '.pt', weights_only=False)
        except:
            print('Model does not exist. Train the model first.')
            sys.exit(0)
        lam1.load_state_dict(checkpoint['lam1'])
        col1.load_state_dict(checkpoint['col1'])
        cub1.load_state_dict(checkpoint['cub1'])
        lam2.load_state_dict(checkpoint['lam2'])
        col2.load_state_dict(checkpoint['col2'])
        cub2.load_state_dict(checkpoint['cub2'])
        b_net.load_state_dict(checkpoint['b_net'])
        v_net.load_state_dict(checkpoint['v_net'])
        kT_net.load_state_dict(checkpoint['kT_net'])

        test_loss = 0
       # Testing step
        for batch in test_data_loader:
            test_loss_batch = test(
                batch, lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, direction, sigma_scaler, device, plot=True
            )
            test_loss += test_loss_batch / len(test_data_loader)
        print('Average MAPE:', test_loss)


    elif sys.argv[1] == 'opt':

        from datetime import datetime

        os.makedirs('opt/', exist_ok=True)

        # Get all trained forward models for optimization
        models = get_all_models()

        # Initialize results DataFrame to store optimization outcomes
        results = pd.DataFrame(columns=[
            'query',                # The queried theta (ground truth)
            'direction',            # The direction case ('X''Y''Z')
            'predicted theta',      # Predicted theta values
            'MAPE',                 # Mean Absolute Percentage Error
            'Query sigma factor',   # Scaling factor for the sigma values
            'predicted direction'   # Predicted direction case
        ])

        # Initialize sigmas DataFrame to store target and reconstructed sigma values
        sigmas = pd.DataFrame(columns=['target_sigma'] * time_steps + ['recon_sigma'] * time_steps)

        # Iterate over each batch in the training data loader
        for batch in tqdm(train_data_loader):
            # Extract the queried theta from the batch and multiply by 70 to unnormalize
            queried_theta = batch[0][0, :] * 70

            # Lists to store losses, theta predictions, and sigma predictions for each direction case
            direction_case_losses = []
            theta_predictions_direction_case = []
            sigma_predictions_all = []

            # Loop over each forward model for evaluation
            for fwd_model in models:
                # Perform optimization with varying numbers of theta guesses (1 to 3)
                for num_thetas in range(1, 4):
                    # Initialize theta guesses and the optimizer
                    theta_guesses, optimizer = init_guess(num_thetas, 20, seed=datetime.now().timestamp())

                    # Perform inverse optimization to predict theta and compute the loss
                    theta_predictions, loss, loss_history, sigma_predictions = optimization(
                        batch, fwd_model, theta_guesses, optimizer, factor=query_factor, direction=direction
                    )

                    # Get the index of the minimum loss in the current optimization run
                    idx = loss.index(min(loss))

                    # Store the minimum loss, corresponding theta prediction, and sigma prediction
                    direction_case_losses.append(loss[idx])
                    theta_predictions_direction_case.append((theta_predictions[idx].detach()[0] * 70).tolist())
                    sigma_predictions_all.append(sigma_predictions[idx])

            # Identify the index of the overall minimum loss across all models and theta guesses
            idx_all = direction_case_losses.index(min(direction_case_losses))

            # Construct a row for the results DataFrame with the queried theta, predicted values, and errors
            res_row = [
                queried_theta.detach().tolist(),            # Ground truth theta
                direction,                                 # Training case label (direction)
                theta_predictions_direction_case[idx_all],  # Predicted theta with the lowest loss
                direction_case_losses[idx_all],             # Corresponding minimum loss (MAPE)
                query_factor,                               # Query sigma scaling factor
                get_direction_case(idx_all)                # Predicted direction case
            ]

            results.loc[len(results)] = res_row  # Add the row to the results DataFrame

            # Prepare a row for the sigmas DataFrame with target and reconstructed sigma values
            sigmas_row = []
            target_sigma = (batch[2][:,get_direction_index(direction)] * query_factor).view(1, -1).detach().tolist()[0]  # Scaled target sigma
            recon_sigma = sigma_predictions_all[idx_all].view(1, -1).detach().tolist()[0]  # Reconstructed sigma
            sigmas_row.extend(target_sigma)
            sigmas_row.extend(recon_sigma)
            sigmas.loc[len(sigmas)] = sigmas_row  # Add the row to the sigmas DataFrame

            # Save the results and sigma values to CSV files
            results.to_csv(f'opt/opt_res_{direction}_factor={query_factor}_thetas.csv')
            sigmas.to_csv(f'opt/opt_res_{direction}_factor={query_factor}_sigmas.csv')
