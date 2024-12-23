import sys
sys.path.insert(0, "../")
from core.__importList__ import *
from core.dataLoader import *
from drivers.model import *
from .config import *

def get_direction_index(direction):
    """Return the index corresponding to the given direction."""
    return {'X': 0, 'Y': 1, 'Z': 2}[direction]

def get_device_tensor(val, device):
    """Return a tensor with a specified value on the given device."""
    return torch.tensor([val]).float().to(device)

def compute_filters(theta, zero_val):
    """Compute the filters for cubic, columnar, and lamellar models based on theta."""
    filter_cub = torch.heaviside(theta[:, 0], zero_val) * torch.heaviside(theta[:, 1], zero_val)
    filter_col = (1 - torch.heaviside(theta[:, 0], zero_val)) * torch.heaviside(theta[:, 1], zero_val)
    filter_lam = (1 - torch.heaviside(theta[:, 0], zero_val)) * (1 - torch.heaviside(theta[:, 1], zero_val))
    return filter_cub, filter_col, filter_lam

def compute_predictions(eps, theta, filter_cub, filter_col, filter_lam, cub_model, col_model, lam_model):
    """Compute the predictions for the given strain, theta, and models."""
    return (
        filter_cub.view(-1, 1) * cub_model(eps, theta) +
        filter_col.view(-1, 1) * col_model(eps, theta[:, 1:3]) +
        filter_lam.view(-1, 1) * lam_model(eps, theta[:, 2:3])
    )

def predict(batch, lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, optimizer, device, mode='train', models=None, theta_pred_3=None, direction=None, query_factor=1.2):
    """Predict function for training, optersion, and testing modes."""
    idx = get_direction_index(direction)
    zero_val = get_device_tensor(0., device)

    # Get batch data
    theta, eps_t, sigma = (batch[0].to(device), batch[1].to(device), batch[2].to(device))
    eps_t.requires_grad = True
    eps_0 = torch.zeros_like(eps_t).to(device).float().requires_grad_(True)

    # Compute 'b' and 'v' based on the mode
    if mode == 'opt':
        b = torch.nn.functional.relu(models[6](theta_pred_3))
        v = torch.nn.functional.relu(models[7](theta_pred_3))
        filter_cub, filter_col, filter_lam = compute_filters(theta_pred_3, zero_val)
    else:
        b = torch.nn.functional.relu(b_net(theta))
        v = torch.nn.functional.relu(v_net(theta))
        filter_cub, filter_col, filter_lam = compute_filters(theta, zero_val) # Compute filters


    # Model predictions
    if mode == 'opt':
        P1_t_pred = models[0](eps_t, theta)
        P1_0_pred = models[0](eps_0, theta)
        P1_b_pred = models[0](b, theta)
        P2_t_pred = models[1]((eps_t - b).float(), theta)
        P2_0_pred = models[1](eps_0, theta)
    else:
        P1_t_pred = compute_predictions(eps_t, theta, filter_cub, filter_col, filter_lam, cub1, col1, lam1)
        P1_0_pred = compute_predictions(eps_0, theta, filter_cub, filter_col, filter_lam, cub1, col1, lam1)
        P1_b_pred = compute_predictions(b, theta, filter_cub, filter_col, filter_lam, cub1, col1, lam1)
        P2_t_pred = compute_predictions((eps_t - b).float(), theta, filter_cub, filter_col, filter_lam, cub2, col2, lam2)
        P2_0_pred = compute_predictions(eps_0, theta, filter_cub, filter_col, filter_lam, cub2, col2, lam2)

    # Compute psi and gradients
    psi1, psi2 = -P1_0_pred, v - P2_0_pred
    H1 = -torch.autograd.grad(P1_0_pred, eps_0, torch.ones(eps_0.shape[0], 1).to(device), create_graph=True)[0]
    H2 = -torch.autograd.grad(P2_0_pred, eps_0, torch.ones(eps_0.shape[0], 1).to(device), create_graph=True)[0]
    dP1de = torch.autograd.grad(P1_t_pred, eps_t, torch.ones(eps_t.shape[0], 1).to(device), create_graph=True)[0]
    dP2de = torch.autograd.grad(P2_t_pred, eps_t, torch.ones(eps_t.shape[0], 1).to(device), create_graph=True)[0]
    dW1de = dP1de + H1
    dW2de = dP2de + H2

    # Compute coefficients and predicted stress
    kT = 0.7 #torch.nn.functional.relu(kT_net(theta))
    W1_t_pred = P1_t_pred + psi1 + H1 * eps_t
    W1_b_pred = P1_b_pred + psi1 + H1 * b
    W2_t_pred = P2_t_pred + psi2 + H2 * (eps_t - b) + W1_b_pred
    W1_coeff = torch.exp(-W1_t_pred / kT) / (torch.exp(-W1_t_pred / kT) + torch.exp(-W2_t_pred / kT))
    W2_coeff = torch.exp(-W2_t_pred / kT) / (torch.exp(-W1_t_pred / kT) + torch.exp(-W2_t_pred / kT))
    sigma_pred = W1_coeff * (dP1de + H1) + W2_coeff * (dP2de + H2)

    # Handle different modes
    
    if mode == 'train':
        loss = lossFn(sigma[:, idx].view(-1, 1), sigma_pred.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, optimizer, loss.item()
    elif mode == 'opt':
        loss = lossFn(query_factor*sigma[:, idx].view(-1, 1), sigma_pred.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item(), sigma_pred, dW1de, dW2de
    elif mode == 'test':
        loss = lossFn(sigma[:, idx].view(-1, 1), sigma_pred.view(-1, 1))
        return loss.item(), sigma_pred, dW1de, dW2de



def test(batch, lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, direction, sigma_scaler, device, plot=True):
    
    idx = get_direction_index(direction)

    loss_batch, sigma_pred, dW1de, dW2de = \
        predict(batch, lam1, col1, cub1, lam2, col2, cub2, b_net, v_net, kT_net, optimizer=None, device=device, mode='test', direction=direction)


    if plot == False:
        return loss_batch

    else:
        plt.rcParams['axes.linewidth'] = 1.5
        theta = batch[0].to(device)
        sigma = batch[2].to(device)[:,idx]
        sigma_pred = sigma_scaler.unnormalize(sigma_pred.view(-1,1))
        sigma = sigma_scaler.unnormalize(sigma.view(-1,1)).view(-1,1)
        dW1de = sigma_scaler.unnormalize(dW1de.view(-1,1))
        dW2de = sigma_scaler.unnormalize(dW2de.view(-1,1))
        eps = torch.linspace(0.0,max_eps,time_steps)           
        Es=3159.93
        os.makedirs('figures/test/',exist_ok=True)
        theta_string = '-'.join([str(int(70*elem)) for elem in theta[0]])
        fig, ax1 = plt.subplots()
        ax1.set_xlabel(r'Strain, $\varepsilon$ [-]',fontsize=21)
        ax1.set_ylabel(r'Norm. Stress, $\sigma/E_S$ [-]',fontsize=21)
        ax1.plot(eps,sigma.view(-1).detach().numpy()/Es,lw=4,label='True',color='#8d342c')
        ax1.plot(eps,sigma_pred.view(-1).detach().numpy()/Es,lw=4,label='Predicted',color='#2c858d')
        ylim = [0,0.006] #ax1.get_ylim()
        ax1.plot(eps,dW1de.view(-1).detach().numpy()/Es,lw=1.5,label='dW1',linestyle='dashed',color='#004056')
        ax1.plot(eps,dW2de.view(-1).detach().numpy()/Es,lw=1.5,label='dW2',linestyle='dashed',color='#2bc9ff')
        ax1.text(0.1*0.4, 0.85*ylim[1],"MAPE = "+"{:.1f}".format(loss_batch*100)+"%", fontsize=18)
        ax1.tick_params(axis='y')#,labelcolor=color
        ax1.set_xticks([0,0.2,0.4])
        ax1.set_yticks([0,0.002,0.004,0.006])
        ax1.set_xticklabels([0,0.2,0.4],fontsize=21)
        ax1.set_ylim(ylim)
        ax1.set_xlim([0,0.4])
        ax1.set_aspect(0.4/ylim[1])
        ax1.minorticks_on()
        plt.subplots_adjust(bottom=0.15)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
        ax1.tick_params(axis='both', which='major', labelsize=21)
        ax1.yaxis.offsetText.set_fontsize(21) 
        plt.savefig('figures/test/'+direction+'_'+theta_string+'.png',dpi=350, transparent=True)
        plt.close()

        return loss_batch


def get_models():

    lam1 = PICNN(n_input=1+1, n_hidden=n_hidden_lam, n_output=1).to(device)
    lam2 = PICNN(n_input=1+1, n_hidden=n_hidden_lam, n_output=1).to(device)
    col1 = PICNN(n_input=2+1, n_hidden=n_hidden_col, n_output=1).to(device)
    col2 = PICNN(n_input=2+1, n_hidden=n_hidden_col, n_output=1).to(device)
    cub1 = PICNN(n_input=3+1, n_hidden=n_hidden_cub, n_output=1).to(device)
    cub2 = PICNN(n_input=3+1, n_hidden=n_hidden_cub, n_output=1).to(device)
    b_net = FFNN(n_input=3,n_hidden=bv_hid, n_output=1).to(device)
    v_net = FFNN(n_input=3,n_hidden=bv_hid, n_output=1).to(device)
    models = [lam1, lam2, col1, col2, cub1, cub2, b_net, v_net]

    return models


def init_guess(num_theta, batch_size, seed):
    """
    Initialize theta guesses and an optimizer for opterse optimization.

    Args:
        num_theta (int): The number of theta parameters to guess.
        batch_size (int): The number of guesses to generate.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A list of theta guesses and an Adam optimizer.
    """
    # Set the random seed for reproducibility
    torch.manual_seed(seed)

    # Create a list to store the theta guesses, each with requires_grad=True
    theta_guesses = [
        torch.randn((1, num_theta), requires_grad=True) for _ in range(batch_size)
    ]

    # Initialize the Adam optimizer for the theta guesses
    optimizer = torch.optim.Adam(theta_guesses, lr=opt_lr)

    # Return the theta guesses and the optimizer
    return theta_guesses, optimizer


def get_all_models():
    model_X = get_models()
    model_Y = get_models()
    model_Z = get_models()

    checkpoint_X = torch.load('models/X.pt', weights_only=False)
    model_X[0].load_state_dict(checkpoint_X['lam1'])
    model_X[1].load_state_dict(checkpoint_X['lam2'])
    model_X[2].load_state_dict(checkpoint_X['col1'])
    model_X[3].load_state_dict(checkpoint_X['col2'])
    model_X[4].load_state_dict(checkpoint_X['cub1'])
    model_X[5].load_state_dict(checkpoint_X['cub2'])
    model_X[6].load_state_dict(checkpoint_X['b_net'])
    model_X[7].load_state_dict(checkpoint_X['v_net'])

    checkpoint_Y = torch.load('models/Y.pt', weights_only=False)
    model_Y[0].load_state_dict(checkpoint_Y['lam1'])
    model_Y[1].load_state_dict(checkpoint_Y['lam2'])
    model_Y[2].load_state_dict(checkpoint_Y['col1'])
    model_Y[3].load_state_dict(checkpoint_Y['col2'])
    model_Y[4].load_state_dict(checkpoint_Y['cub1'])
    model_Y[5].load_state_dict(checkpoint_Y['cub2'])
    model_Y[6].load_state_dict(checkpoint_Y['b_net'])
    model_Y[7].load_state_dict(checkpoint_Y['v_net'])

    checkpoint_Z = torch.load('models/Z.pt', weights_only=False)
    model_Z[0].load_state_dict(checkpoint_Z['lam1'])
    model_Z[1].load_state_dict(checkpoint_Z['lam2'])
    model_Z[2].load_state_dict(checkpoint_Z['col1'])
    model_Z[3].load_state_dict(checkpoint_Z['col2'])
    model_Z[4].load_state_dict(checkpoint_Z['cub1'])
    model_Z[5].load_state_dict(checkpoint_Z['cub2'])
    model_Z[6].load_state_dict(checkpoint_Z['b_net'])
    model_Z[7].load_state_dict(checkpoint_Z['v_net'])


    all_models = [model_X,model_Y,model_Z]

    return all_models



def optimization(batch, model, theta_guesses, optimizer, factor=1.2, direction=None):
    """
    Perform opterse optimization to predict theta values by minimizing the loss between predicted and target stress values.

    Args:
        batch (list): The input batch containing strain-stress data.
        model (list): List of forward models used for prediction.
        theta_guesses (list): Initial guesses for theta values, each with gradients enabled.
        optimizer (torch.optim.Optimizer): Optimizer for updating theta guesses.
        factor (float): Scaling factor for the stress values (default: 1.2).

    Returns:
        tuple: A tuple containing:
            - theta_predictions (list): List of optimized theta predictions.
            - losses (list): List of final loss values for each theta guess.
            - loss_history (list): Empty list (placeholder for loss history if needed).
            - sigma_predictions (list): List of predicted stress values (sigma).
    """
    # Initialize empty lists to store results
    loss_history = []
    theta_predictions = []
    losses = []
    sigma_predictions = []

    # Iterate over each initial theta guess
    for theta_guess in theta_guesses:
        for _ in tqdm(range(opt_epochs), desc="Optimizing Theta"):
            # Apply sigmoid to constrain theta values between 0 and 1
            theta_guess_in = torch.sigmoid(theta_guess).float()

            # Sort theta values to maintain order
            theta_guess_in, _ = torch.sort(theta_guess_in)

            # Enforce minimum constraint (20/70) on theta values based on the number of thetas
            for i in range(theta_guess_in.shape[1]):
                if theta_guess_in[0, i] < 20.0 / 70.0:
                    theta_guess_in[0, i] = 20.0 / 70.0

            # Update the batch with the constrained theta guess
            batch[0] = theta_guess_in

            # Prepare theta_pred_3 based on the number of thetas
            num_thetas = theta_guess_in.shape[1]
            zero = torch.zeros(1, 1).float()
            if num_thetas == 1:
                theta_pred_3 = torch.cat((zero, zero, theta_guess_in), dim=1)
            elif num_thetas == 2:
                theta_pred_3 = torch.cat((zero, theta_guess_in), dim=1)
            else:
                theta_pred_3 = theta_guess_in

            # Select appropriate forward models based on the number of thetas
            models = model[(num_thetas - 1) * 2:num_thetas * 2]
            models.extend([None, None, None, None])
            models.extend([model[6], model[7]])

            # Perform forward prediction and compute the loss
            loss, sigma_pred, _, _ = predict(
                batch, None, None, None, None, None, None, None, None, None, \
                optimizer, device, mode='opt', models=models, \
                theta_pred_3=theta_pred_3, direction=direction, query_factor=factor
            )

        # Store results after optimization
        sigma_predictions.append(sigma_pred.detach())
        theta_predictions.append(theta_pred_3)
        losses.append(loss)

    print('--------------------------------------------------------------------------------------')
    return theta_predictions, losses, loss_history, sigma_predictions
