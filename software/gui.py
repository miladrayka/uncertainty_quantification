import math
import json

import streamlit as st
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


st.set_page_config(
    page_title="FFNN-ECIF-BayeByBackprop",
    page_icon=r"title_logo.png",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("Uncertainty Prediction Software")
st.header("FFNN-ECIF-BayeByBackprop")

st.image(r"Logo.png")

st.sidebar.header("Developer")
st.sidebar.write(
    """[GitHub](https://github.com/miladrayka/uncertainty_quantification),
    Developed by *[Milad Rayka](https://scholar.google.com/citations?user=NxF2f0cAAAAJ&hl=en)*."""
)
st.sidebar.divider()
st.sidebar.header("Citation")
st.sidebar.write(
    """**Reference**:
    Paper is *under production.*"""
)

st.write(
    """FFNN-ECIF-BayeByBackprop is a software for protein-ligand binding affinity prediction and uncertainty quantification.
 The model is based on FeedForward-NeuralNetwork (FFNN) and Extended-Connectivity Interaction Feature (ECIF) for binding affinity prediction.
 We augment our model with Bayes by the Backprop approach for uncertainty quantification of predicted binding affinity."""
)
with st.expander("**Cautions**"):
    st.info(
        "To generate ECIF for a protein-ligand complex, use [REINDEER](https://github.com/miladrayka/reindeer_software)."
    )


path = st.text_input(
    "Enter the path of ECIF:",
    placeholder="Type a path...",
    help="Provide the path of protein-ligand ECIF csv file, e.g., ./data_instances_ecif.csv",
)

DEVICE = (
    torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
)
print(f"Device: {DEVICE.type}")

with open("model_params.json", "r") as fp:
    hyperparameters = json.load(fp)

n_units_l_params = {k: v for k, v in hyperparameters.items() if "n_units_l_" in k}

IN_FEATURES = 790
EPOCHS = 300
BATCH_SIZE = hyperparameters["batch_size"]
LEARNING_RATE = hyperparameters["lr"]
WEIGHT_DECAY = hyperparameters["wd"]
N_LAYERS = hyperparameters["n_layers"]
NUM_ENU_LIST = list(n_units_l_params.values())
P = hyperparameters["dropout"]
OPTIMIZER_NAME = hyperparameters["optimizer"]
NUM_BATCHES = 227  # len(dataloader_dict["LP-Train"])
SAMPLES = 10
# Gaussian mixture parameters
if DEVICE.type == "cpu":
    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])
else:
    PI = 0.5
    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

# Most of the following codes are adopted by some changes from below repository:
# https://github.com/nitarshan/bayes-by-backprop


class Gaussian(object):

    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho

        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):

        return torch.log1p(torch.exp(self.rho))

    def sample(self):

        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)

        return self.mu + self.sigma * epsilon

    def log_prob(self, input):

        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma**2)
        ).sum()


class ScaleMixtureGaussian(object):

    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi

        self.sigma1 = sigma1

        self.sigma2 = sigma2

        self.gaussian1 = torch.distributions.Normal(0, sigma1)

        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):

        prob1 = torch.exp(self.gaussian1.log_prob(input))

        prob2 = torch.exp(self.gaussian2.log_prob(input))

        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2)
        )
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4)
        )
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight
            ) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight
            ) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianLinear(790, 900)
        self.l2 = BayesianLinear(900, 300)
        self.l3 = BayesianLinear(300, 1)

    def forward(self, x, sample=False):
        x = F.elu(self.l1(x, sample))
        x = F.dropout(x, 0.4)
        x = F.elu(self.l2(x, sample))
        x = F.dropout(x, 0.4)
        x = self.l3(x, sample)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return (
            self.l1.log_variational_posterior
            + self.l2.log_variational_posterior
            + self.l3.log_variational_posterior
        )

    def sample_elbo(self, input, target, batch_size, samples=SAMPLES):
        outputs = torch.zeros(samples, batch_size).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True).reshape(-1)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.mse_loss(
            outputs.mean(0), target.reshape(-1), reduction="sum"
        )
        loss = (
            log_variational_posterior - log_prior
        ) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood


net = BayesianNetwork().to(DEVICE)
saved_model_path = f"./model.pth"
net.load_state_dict(
    torch.load(saved_model_path, map_location=torch.device("cpu"))["model_state_dict"]
)
net.to(DEVICE)

try:
    feat_stat_df = pd.read_csv("Remained_features_statistics.csv")
    input_df = pd.read_csv(path)
    names = input_df.iloc[:, 0].to_list()
    scaled_input_df = (
        input_df.loc[:, feat_stat_df.columns] - feat_stat_df.loc[0, :]
    ) / (feat_stat_df.loc[1, :])
    scaled_input_df = scaled_input_df.fillna(0)
    input_array = scaled_input_df.to_numpy()
    data = torch.Tensor(input_array)
except:
    pass

st.write("Push the Run button:")

run = st.button("Run")

if run:

    net.eval()
    with torch.no_grad():
        data = data.to(DEVICE)
        outputs = torch.zeros(len(data), SAMPLES).to(DEVICE)
        for i in range(len(data)):
            for j in range(SAMPLES):
                outputs[i, j] = net(data[i, :].unsqueeze(0), sample=True).reshape(-1)
        output = outputs.mean(1)
        var = outputs.var(1)

    result_df = pd.DataFrame({"names": names, "prediction": output, "variance": var})

    st.info(
        "Prediction and Variance columns are predicted binding affinity and quantified uncertainty, respectively."
    )

    st.dataframe(result_df)
