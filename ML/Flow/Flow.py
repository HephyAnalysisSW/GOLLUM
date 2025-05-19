import os
import glob
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import re
from common import data_structure

from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.nn.nets import ResidualNet
from nflows.flows import Flow as FlowModel
from data_loader.data_loader_2 import H5DataLoader

logger = logging.getLogger('UNC')

class SimpleContextNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, context_features):
        super().__init__()
        dims = [in_features + context_features] + hidden_features + [out_features]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x, context):
        # x: [B, in_features], context: [B, context_features]
        h = torch.cat([x, context], dim=-1)
        return self.net(h)

class SplineNetWithScale(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, context_features, scale_factor):
        super().__init__()
        self.inner = SimpleContextNet(
            in_features=in_features,
            out_features=out_features,
            hidden_features=hidden_features,
            context_features=context_features
        )
        self.scale  = scale_factor

    def forward(self, x, context):
        # x: [B, in_features], context: [B, context_features]
        params = self.inner(x, context)     # [B, out_features]
        return params * self.scale

def build_spline_net_with_zero_bias(in_f, out_f, hidden, ctx_dim, scale):
    net = SplineNetWithScale(
        in_features=in_f,
        out_features=out_f,
        hidden_features=hidden,
        context_features=ctx_dim,
        scale_factor=scale
    )
    # zero‐bias trick:
    for name, param in net.named_parameters():
        if 'bias' in name:
            param.data.zero_()
    return net

class Flow:
    def __init__(self, config=None, model_dir=None):
        if config is None:
            raise ValueError("Please provide a config.")
        self.config    = config
        self.input_dim = config.input_dim
        self.model_dir = model_dir

        # -- embedding net for ν
        nu_dim = len(config.parameters)
        layers = []
        in_dim = nu_dim
        for h in config.embed_hidden_layers:
            layers += [nn.Linear(in_dim, h), eval(config.activation)]
            in_dim = h
        layers += [nn.Linear(in_dim, config.embed_dim), eval(config.activation)]
        self.embed_net = nn.Sequential(*layers)

        # -- build the flow
        transforms = []
        for i in range(config.n_flow_layers):
            # 1) random permutation
            transforms.append(RandomPermutation(features=self.input_dim))

            # 2) build an alternating base mask
            idx       = torch.arange(self.input_dim)
            base_mask = ((idx % 2) == (i % 2))  # True/False alternating, offset by i

            # 3) two coupling blocks: checkerboard and its inverse
            for mask in (base_mask, ~base_mask):
                if config.use_spline:
                    transforms.append(
                        PiecewiseRationalQuadraticCouplingTransform(
                            mask=mask.to(torch.bool),
                            #transform_net_create_fn=(
                            #    lambda in_f, out_f: SplineNetWithScale(
                            #        in_features=in_f,
                            #        out_features=out_f,
                            #        hidden_features=config.flow_hidden_layers,
                            #        context_features=config.embed_dim,
                            #        scale_factor = 1.0 / (config.num_bins * config.bin_range)

                            #    )
                            #),
                            transform_net_create_fn=lambda in_f, out_f: build_spline_net_with_zero_bias(
                                in_f, out_f,
                                hidden=config.flow_hidden_layers,
                                ctx_dim=config.embed_dim,
                                scale=1.0 / config.num_bins
                            ),
                            num_bins=config.num_bins,
                            tail_bound=config.bin_range,
                            # you can add tails='linear' or 'circular' if needed
                            tails='linear',            # <— allow linear tails
                            min_bin_width=config.min_bin_width,        # optional stability tweaks
                            min_bin_height=config.min_bin_height,
                            min_derivative=config.min_derivative,
                        )
                    )
                else:
                    transforms.append(
                        AffineCouplingTransform(
                            mask=mask.to(torch.bool),
                            transform_net_create_fn=lambda in_f, out_f:
                                SimpleContextNet(
                                    in_features=in_f,
                                    out_features=out_f,
                                    hidden_features=config.flow_hidden_layers,
                                    context_features=config.embed_dim
                                ),
                        )
                    )

            # 4) optional batch‐norm
            if getattr(config, "use_batch_norm", False):
                transforms.append(nflows.transforms.BatchNorm(features=self.input_dim))

        self.transform = CompositeTransform(transforms)
        base = StandardNormal([self.input_dim])
        self.model = FlowModel(
            transform=self.transform,
            distribution=base
        )
        # feature‐scaling
        if hasattr(config, "feature_means"):
            self.feature_means     = config.feature_means
            self.feature_variances = config.feature_variances
        else:
            D = len(data_structure.feature_names)
            self.feature_means     = np.zeros(D, dtype=float)
            self.feature_variances = np.ones(D,  dtype=float)

    def load_training_data( self, datasets_hephy, selection, process=None, n_split=10):
        self.training_data = {}
        self.process = process
        self.n_split = n_split
        for base_point in self.config.base_points:
            base_point = tuple(base_point)
            values = self.config.get_alpha(base_point)
            data_loader = datasets_hephy.get_data_loader( selection=selection, values=values, process=process, selection_function=None, n_split=n_split)
            logger.info ("Flow training data: process %s Base point nu = %r, alpha = %r, file = %s"%( (process if process is not None else "combined"), base_point, values, data_loader.file_path))
            self.training_data[base_point] = data_loader

    @classmethod
    def load(cls, model_dir, config):
        """
        Load the most recent Flow checkpoint from `model_dir`.
        If `flow_final.pt` exists, it is used; otherwise the highest‐numbered
        `flow_epoch_*.pt` is selected.
        """
        # Determine checkpoint path
        if os.path.isdir(model_dir):
            final_ckpt = os.path.join(model_dir, "flow_final.pt")
            if os.path.exists(final_ckpt):
                ckpt_path = final_ckpt
            else:
                pattern = os.path.join(model_dir, "flow_epoch_*.pt")
                candidates = glob.glob(pattern)
                if not candidates:
                    raise FileNotFoundError(f"No checkpoints found in {model_dir}")
                # Extract epoch numbers and pick the largest
                epochs = []
                for fn in candidates:
                    m = re.match(r".*flow_epoch_(\d+)\.pt$", fn)
                    if m:
                        epochs.append((int(m.group(1)), fn))
                if not epochs:
                    raise FileNotFoundError(f"No valid epoch files in {model_dir}")
                # sort by epoch and take the last
                ckpt_path = sorted(epochs, key=lambda x: x[0])[-1][1]
        else:
            raise FileNotFoundError(f"{model_dir} is not a directory")

        # Load the checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Instantiate and populate the model
        obj = cls(config=config)
        obj.model.load_state_dict(ckpt["model_state"])
        obj.embed_net.load_state_dict(ckpt["embed_state"])
        # feature scaling vectors
        obj.feature_means     = ckpt["feature_means"]
        obj.feature_variances = ckpt["feature_variances"]
        obj.model_dir         = model_dir
        return obj

    def save(self, epoch=None):
        """
        Save model, embedding state, and feature normalization stats.
        If `epoch` is None, writes `flow_final.pt`, else `flow_epoch_{epoch}.pt`.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        if epoch is None:
            fname = "flow_final.pt"
        else:
            fname = f"flow_epoch_{epoch}.pt"
        path = os.path.join(self.model_dir, fname)
        torch.save({
            "model_state": self.model.state_dict(),
            "embed_state": self.embed_net.state_dict(),
            "feature_means": self.feature_means,
            "feature_variances": self.feature_variances
        }, path)


    def forward(self, x, nu):
        """
        Forward pass: map x to latent z under conditioning nu.
        """
        context = self.embed_net(nu)
        z, logdet = self.model._transform.forward((x-self.feature_means)/np.sqrt(self.feature_variances), context=context)
        return z, logdet

    def log_prob(self, x, nu):
        """
        Compute log probability of x given nu, where nu can be tuple/list or Tensor.
        x should be a Tensor of shape [B, input_dim].
        """
        # ensure x on correct device/dtype
        x = x.to(self.device)

        # build nu context
        if not torch.is_tensor(nu):
            nu_vec = torch.tensor(nu, dtype=torch.float32, device=self.device)
        else:
            nu_vec = nu.to(device=self.device, dtype=torch.float32)
        # repeat or expand to [B, nu_dim]
        nu_ctx = nu_vec.unsqueeze(0).expand(x.size(0), -1)

        # normalize x
        means = torch.from_numpy(self.feature_means).to(self.device)
        vars_  = torch.from_numpy(self.feature_variances).to(self.device)
        x_norm = ((x - means) / torch.sqrt(vars_)).to(torch.float32)

        # compute log‐prob
        context = self.embed_net(nu_ctx)
        return self.model.log_prob(x_norm, context=context)

    def sample(self, num_samples, nu):
        """
        Draw samples from the flow given nu, where nu can be a tuple/list or a Tensor.
        """
        # 1) turn nu into a float32 tensor of shape [nu_dim]
        with torch.no_grad():
            if not torch.is_tensor(nu):
                nu_vec = torch.tensor(nu, dtype=torch.float32, device=self.device)
            else:
                nu_vec = nu.to(device=self.device, dtype=torch.float32)

            # 2) build a single‐row context [1, nu_dim]
            nu_ctx = nu_vec.unsqueeze(0)                    # [1, nu_dim]
            # 3) embed that one context
            c = self.embed_net(nu_ctx)                      # [1, embed_dim]
            # 4) draw N samples for that single context: returns [1, N, input_dim]
            samples = self.model.sample(num_samples, context=c)
        # 5) drop the first (context) axis → [N, input_dim] & restore features & variances
        return samples.squeeze(0)*np.sqrt(self.feature_variances) + self.feature_means        

    def train_one_epoch(self, max_batch: int = -1, accumulate_histograms: bool = False):
        """
        Train the flow for one epoch over all ν slices in lockstep.
        If accumulate_histograms=True, also builds:
          - true_hist[feat]: weighted data‐histogram across 7 ν‐points
          - weight_sums[nu]: total event weight at each ν
        Returns:
          (true_hist, weight_sums) or (None, None)
        """
        # filter to those we actually have
        nus = [nu for nu in self.config.nu_plot_list if nu in self.training_data]

        if accumulate_histograms:
            # init true‐hist & bin edges
            true_hist = { nu: {
                feat: np.zeros(data_structure.plot_options[feat]['binning'][0])
                for feat in data_structure.plot_options
            } for nu in nus }

            bin_edges = {
                feat: np.linspace(xmin, xmax, n_bins+1)
                for feat, (n_bins, xmin, xmax)
                in {f:data_structure.plot_options[f]['binning'] for f in data_structure.plot_options}.items()
            }
            # total weights per ν
            weight_sums = {nu: 0.0 for nu in nus}

        # set up iterators for ALL ν (train over full grid)
        data_iters = {nu: iter(loader) for nu, loader in self.training_data.items()}
        n_slices = len(data_iters)

        total_loss = 0.0
        total_samples = 0
        batch_idx = 0

        total_iters = self.n_split
        if 0 < max_batch < total_iters:
            total_iters = max_batch

        means = torch.from_numpy(self.feature_means).to(self.device)
        vars_  = torch.from_numpy(self.feature_variances).to(self.device)

        # Loop with progress bar
        for batch_idx in tqdm(range(total_iters), desc="Training batches"):
            # Try to fetch one batch per ν; if any loader is exhausted, stop early
            try:
                batches = {nu: next(it) for nu, it in data_iters.items()}
            except StopIteration:
                break

            self.optimizer.zero_grad()
            batch_loss = 0.0

            for nu_tuple, batch in batches.items():
                x_batch, weights, _ = H5DataLoader.split(batch)

                # accumulate total weight if this ν in our subset
                if accumulate_histograms and nu_tuple in weight_sums:
                    weight_sums[nu_tuple] += weights.sum()

                # convert weights to float32 tensor
                w = torch.tensor(weights, device=self.device, dtype=torch.float32)

                # prepare x_norm
                x = torch.as_tensor(x_batch, device=self.device, dtype=torch.float32) \
                    if isinstance(x_batch, np.ndarray) else x_batch.to(self.device, torch.float32)

                x_norm = ((x - means) / torch.sqrt(vars_)).to(torch.float32)

                # build context
                nu_vec = torch.tensor(nu_tuple, device=self.device, dtype=torch.float32)
                nu_ctx = nu_vec.unsqueeze(0).expand(x.size(0), -1)  # [B,nu_dim]

                # compute weighted NLL
                logp = self.model.log_prob(x_norm, context=self.embed_net(nu_ctx))
                loss_i = -(w * logp).sum()
                batch_loss += loss_i

                # accumulate true hist only for our 7 ν‐points
                if accumulate_histograms and nu_tuple in nus:
                    for fid, feat in enumerate(data_structure.feature_names):
                        vals = x_batch[:, fid]
                        h_t, _ = np.histogram(vals, bins=bin_edges[feat], weights=weights)
                        true_hist[nu_tuple][feat] += h_t

            # backprop
            batch_loss = batch_loss / n_slices
            batch_loss.backward()
            self.optimizer.step()

            total_loss += batch_loss.item() * x.size(0)
            total_samples += x.size(0)
            batch_idx += 1

        epoch_loss = total_loss / total_samples
        logger.info(f"Epoch loss: {epoch_loss:.4f}")

        if accumulate_histograms:
            return true_hist, weight_sums
        else:
            return None, None


    def plot_convergence_root(self,
                              true_hist: dict,
                              weight_sums: dict,
                              epoch: int,
                              output_path: str,
                              feature_names):
        """
        Plot empirical vs. flow‐generated histograms for the 7 fixed ν‐points.
        - true_hist[feat] holds data‐weighted counts.
        - weight_sums[nu] holds total weight per ν.
        """
        import ROOT
        from math import ceil, sqrt
        ROOT.gStyle.SetOptStat(0)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        ROOT.gROOT.LoadMacro(os.path.join(dir_path, "../../common/scripts/tdrstyle.C"))
        ROOT.setTDRStyle()

        os.makedirs(output_path, exist_ok=True)

        # recompute bin edges
        bin_edges = {
            feat: np.linspace(xmin, xmax, n_bins+1)
            for feat, (n_bins, xmin, xmax)
            in {f:data_structure.plot_options[f]['binning'] for f in feature_names}.items()
        }

        # filter
        nus = [nu for nu in self.config.nu_plot_list if nu in weight_sums]

        # build model histograms by sampling
        pred_hist = {nu: {feat: np.zeros_like(true_hist[nu][feat]) for feat in feature_names} for nu in nus}

        for nu_tuple in nus:
            n_pred = int(round(weight_sums[nu_tuple])) or 1
            # sample from flow
            samples = self.sample(n_pred, nu_tuple).detach().cpu().numpy()
            for fid, feat in enumerate(feature_names):
                vals = samples[:, fid]
                h_p, _ = np.histogram(vals, bins=bin_edges[feat])
                pred_hist[nu_tuple][feat] += h_p

        # now plot just like before
        num_features = len(feature_names)
        for normalized in (False, True):
            if normalized:
                for feat in feature_names:
                    for nu in nus:
                        total = true_hist[nu][feat].sum()
                        if total>0:
                            true_hist[nu][feat] /= total
                            pred_hist[nu][feat] /= total

            total_pads = num_features + 1
            gx = int(ceil(sqrt(total_pads)))
            gy = int(ceil(total_pads/gx))
            canvas = ROOT.TCanvas("c", "conv", 500*gx, 500*gy)
            canvas.Divide(gx, gy)

            colors = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen, ROOT.kOrange, ROOT.kMagenta]
            stuff = []

            for i, feat in enumerate(feature_names):
                pad = canvas.cd(i+1)
                pad.SetTicks(1,1)
                pad.SetBottomMargin(0.15)
                pad.SetLeftMargin(0.15)
                pad.SetLogy(not normalized and data_structure.plot_options[feat]['logY'])

                max_y = max( [max(true_hist[nu][feat].max(), pred_hist[nu][feat].max()) for nu in nus] )
                n_bins, x_min, x_max = data_structure.plot_options[feat]['binning']
                x_title = data_structure.plot_options[feat]['tex']

                h_frame = ROOT.TH2F(f"f_{feat}",
                                    f";{x_title};density",
                                    n_bins, x_min, x_max,
                                    100, 0, 1.2*max_y)
                h_frame.GetYaxis().SetTitleOffset(1.3)
                h_frame.Draw()
                stuff.append(h_frame)

                for i_nu, nu in enumerate(nus):
                    # empirical
                    h_t = ROOT.TH1F(f"t_{feat}", "", n_bins, x_min, x_max)
                    for b in range(n_bins):
                        h_t.SetBinContent(b+1, true_hist[nu][feat][b])
                    h_t.SetLineColor(colors[i_nu]); h_t.SetLineStyle(2); h_t.SetLineWidth(2)
                    h_t.Draw("HIST SAME"); stuff.append(h_t)

                    # model
                    h_p = ROOT.TH1F(f"p_{feat}", "", n_bins, x_min, x_max)
                    for b in range(n_bins):
                        h_p.SetBinContent(b+1, pred_hist[nu][feat][b])
                    h_p.SetLineColor(colors[i_nu]); h_p.SetLineStyle(1); h_p.SetLineWidth(2)
                    h_p.Draw("HIST SAME"); stuff.append(h_p)

            # legend pad
            pad = canvas.cd(num_features+1)
            legend = ROOT.TLegend(0.1,0.1,0.9,0.9)
            legend.SetBorderSize(0); legend.SetShadowColor(0)
            for i_nu, nu in enumerate(nus):
                d_t = ROOT.TH1F("dt","",1,0,1); d_p = ROOT.TH1F("dp","",1,0,1)
                d_t.SetLineColor(colors[i_nu]); d_t.SetLineStyle(2); d_t.SetLineWidth(2)
                d_p.SetLineColor(colors[i_nu]); d_p.SetLineStyle(1); d_p.SetLineWidth(2)
                legend.AddEntry(d_t, "true %r"%list(nu), "l")
                legend.AddEntry(d_p, "pred %r"%list(nu), "l")
            legend.Draw(); stuff.extend([d_t, d_p])

            tex = ROOT.TLatex(); tex.SetNDC(); tex.SetTextSize(0.07); tex.SetTextAlign(11)
            tex.DrawLatex(0.3, 0.95, f"Epoch = {epoch:5d}")
            canvas.Update()

            suffix = "" if not normalized else "norm_"
            outfile = os.path.join(output_path, f"{suffix}epoch_{epoch:04d}.png")
            canvas.SaveAs(outfile)
            logger.info(f"Saved {outfile}")

