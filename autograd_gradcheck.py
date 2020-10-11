import torch
import copy
from networks import debug_networks
import hamiltonian_generative_network
from utils import integrator


def compute(phi, w, gamma, theta):
    """Compute the HGN loss from the given weights for a fixed input.
    """

    # Instantiate networks
    encoder = debug_networks.EncoderNet(phi=phi, seq_len=ROLLOUTS.shape[1], dtype=DTYPE)
    transformer = debug_networks.TransformerNet(w=w, dtype=DTYPE)
    hnn = debug_networks.HamiltonianNet(gamma=gamma, dtype=DTYPE)
    decoder = debug_networks.DecoderNet(theta=theta, dtype=DTYPE)

    # Define HGN integrator
    integr = integrator.Integrator(delta_t=0.01, method="Euler")

    # Define optimization module
    optim_params = [
        {'params': encoder.parameters(), },
        {'params': transformer.parameters(), },
        {'params': hnn.parameters(), },
        {'params': decoder.parameters(), },
    ]
    optimizer = torch.optim.SGD(optim_params, lr=0.01, momentum=0.9)
    loss = torch.nn.MSELoss()

    # Instantiate Hamiltonian Generative Network
    hgn = hamiltonian_generative_network.HGN(
        encoder=encoder,
        transformer=transformer,
        hnn=hnn,
        decoder=decoder,
        integrator=integr,
        loss=loss,
        optimizer=optimizer,
        seq_len=ROLLOUTS.shape[1],
        channels=1
    )

    error = hgn.fit(ROLLOUTS)
    return error


if __name__ == '__main__':
    torch.manual_seed(10)
    DTYPE = torch.double
    BATCH_SIZE = 1  # TODO: Not working!!
    SEQ_LEN = 100

    # Generate random inputs
    ROLLOUTS = torch.randn((BATCH_SIZE, SEQ_LEN, 1), dtype=DTYPE)

    # Generate random parameters (to be perturbed)
    phi = torch.randn(2, requires_grad=True, dtype=DTYPE)
    w = torch.randn(2, requires_grad=True, dtype=DTYPE)
    gamma = torch.randn(2, requires_grad=True, dtype=DTYPE)
    theta = torch.randn(1, requires_grad=True, dtype=DTYPE)

    correct = torch.autograd.gradcheck(
        compute,
        inputs=(phi, w, gamma, theta),
        #eps=1e-5.,  # Perturbation applied to the parameters (why smaller is harder?)
        #rtol=1e-5  # Relative tolerance
    )

    print('Correct? ' + str(correct))