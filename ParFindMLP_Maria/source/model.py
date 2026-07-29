import torch
import torch.nn as nn
import lindblad_solver as solver


# MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation='relu', use_batchnorm=False, dropout_rate=0.0, output_activation='linear'):

        super().__init__()              # run the parent class constructor (yeah, i'm just a physicist ...)

        self.input_dim = input_dim                      # input dimensionality
        self.output_dim = output_dim                    # output dimensionality
        self.hidden_dims = hidden_dims                  # list of hidden layer sizes
        self.activation = activation                    # activation function to use
        self.use_batchnorm = use_batchnorm              # use batch normalization?
        self.dropout_rate = dropout_rate                # dropout rate
        self.output_activation = output_activation      # activation function to use on the output

        self.model = self._build_model()                # build the model

    def _make_activation(self):
        if self.activation == 'relu':
            return nn.ReLU()
        elif self.activation == 'tanh':
            return nn.Tanh()
        elif self.activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {self.activation}. Must be one of 'relu', 'tanh', or 'leaky_relu'.")

    def _init_weights(self, linear_layer):
        # Match initialization to the activation that follows this layer.
        if self.activation == 'relu':
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='relu')
        elif self.activation == 'leaky_relu':
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif self.activation == 'tanh':
            nn.init.xavier_uniform_(linear_layer.weight)
        else:
            raise ValueError(f"Unknown activation: {self.activation}. Must be one of 'relu', 'tanh', or 'leaky_relu'.")
        nn.init.zeros_(linear_layer.bias)

    def _build_model(self):
        layers = []
        layer_sizes = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]

        # Hidden layers: Linear -> (BatchNorm) -> Activation -> (Dropout)
        for i in range(len(layer_sizes) - 2):
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self._init_weights(linear_layer)
            layers.append(linear_layer)

            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

            layers.append(self._make_activation())

            if self.dropout_rate > 0.0:
                layers.append(nn.Dropout(p=self.dropout_rate))

        output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        if self.output_activation == 'linear':
            pass  # no activation -- correct default for regression
        elif self.output_activation == 'tanh':
            layers.append(nn.Tanh())
        elif self.output_activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}. Must be one of 'linear', 'tanh', or 'leaky_relu'.")

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# PINN model

class PINN(nn.Module):

    def __init__(self, feature_dim, H, L_ops, dt, hidden_dims, activation='relu', use_batchnorm=False, dropout_rate=0.0, output_activation='linear'):
        """
        :param:
            feature_dim: input dimensionality (from TrajectoryDataset.feature_dim)
            H: (N, N) complex array-like Hamiltonian, fixed/known (not learned)
            L_ops: list of (N, N) complex array-like jump operators, fixed/known
            dt: float, integration time step (must match the dataset's dt)
            hidden_dims, activation, use_batchnorm, dropout_rate: passed through to MLP
        """
        super().__init__()

        self.mlp = MLP(
            input_dim=feature_dim,
            output_dim=5,  # 5 tau parameters
            hidden_dims=hidden_dims,
            activation=activation,
            use_batchnorm=use_batchnorm,
            dropout_rate=dropout_rate,
            output_activation=output_activation,  # predicting log-tau, unconstrained
        )

        placeholder_gammas = [1.0] * len(L_ops) #  gammas here are placeholders -- step_batched() always receives the actual per-sample predicted gammas at call time
        self.solver = solver.Lindblad_solver(H, L_ops, placeholder_gammas, dt)

    def forward(self, features):
        """
        :param:
            features: (B, feature_dim) float tensor
        :return:
            taus_pred_log : (B, 5) float tensor, predicted log(tau) - network's raw output
            taus_pred     : (B, 5) float tensor, predicted tau (physical units, undoing the log)
            gammas_pred   : (B, 5) float tensor, predicted gamma_k = 1 / tau_k
        """
        taus_pred_log = self.mlp(features)
        taus_pred = torch.exp(taus_pred_log)  # undo dataset.py's log label_transform
        gammas_pred = 1.0 / taus_pred

        return taus_pred_log, taus_pred, gammas_pred

    def data_loss(self, taus_pred_log, labels_log):
        """
        MSE between predicted and ground-truth log-taus.

        :param:
            taus_pred_log: (B, 5) float tensor
            labels_log:    (B, 5) float tensor
        :return:
            scalar tensor
        """
        return torch.mean((taus_pred_log - labels_log) ** 2)

    def physics_loss(self, rho_t, rho_tp1, gammas_pred):
        """
        Single-step Lindblad-residual loss.

        :param:
            rho_t: (B, N, N) complex tensor, known state at time t
            rho_tp1:     (B, N, N) complex tensor, ground-truth state at t+dt
            gammas_pred: (B, 5) float tensor, predicted decay/dephasing rates
        :return:
            scalar tensor
        """

        B, K = rho_t.shape[:2]

        rho_pred = self.solver.step_batched(
            rho_t.reshape(B * K, 4, 4),
            gammas_pred.repeat_interleave(K, dim=0) * 1e12
        )

        rho_pred = rho_pred.reshape(B, K, 4, 4)

        diff = rho_pred - rho_tp1

        return torch.mean(diff.abs() ** 2)

    def total_loss(self, batch, lambda_phys=1.0, lambda_data=1.0):
        """
        Combined data + physics loss for one training batch.

        :param:
            batch: tuple (features, labels, rho_t, rho_tp1) as returned by TrajectoryDataset
            lambda_phys: float, weight on the physics loss term
        :return:
            loss      : scalar tensor, the combined total loss (for backward())
            loss_data : scalar tensor, data loss component (for logging)
            loss_phys : scalar tensor, physics loss component (for logging)
        """
        features, labels_log, rho_t, rho_tp1 = batch

        taus_pred_log, taus_pred, gammas_pred = self.forward(features)

        loss_data = self.data_loss(taus_pred_log, labels_log)
        loss_phys = self.physics_loss(rho_t, rho_tp1, gammas_pred)

        #print(f"labels log: {labels_log}, pred log {taus_pred_log}")

        loss = lambda_data * loss_data + lambda_phys * loss_phys

        return loss, loss_data, loss_phys


