class HgnResult():
    """Holds HGN evolution guess information
    """

    def __init__(self):
        self.z_mean = None
        self.z_std = None
        self.z_sample = None
        self.s_0 = None
        self.q_s = []
        self.p_s = []

    def set_z(self, z_mean, z_std, z_sample):
        """Store latent variable conditions

        Args:
            z_mean (torch.Tensor): Mean of q_z
            z_std (torch.Tensor): Standard dev of q_z
            z_sample (torch.Tensor): Sample taken from q_z distribution
        """
        self.z_mean = z_mean
        self.z_std = z_std
        self.z_sample = z_sample

    def set_initial_state(self, s_0):
        """Store initial state s_0

        Args:
            s_0 (torch.Tensor): Initial state s_0 = [q_0, p_0]
        """
        self.s_0 = s_0

    def add_step(self, q, p):
        """Append the guessed position (q) and momentum (p) to guessed list 

        Args:
            q (torch.Tensor): Position encoded information
            p (torch.Tensor): Momentum encoded information
        """
        self.q_s.append(q)
        self.p_s.append(p)