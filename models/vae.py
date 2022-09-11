import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.utils import loss_functions as lf, modules
from models.conv.nets import ConvLayers,DeconvLayers
from models.fc.nets import MLP, MLP_gates
from models.fc.layers import fc_layer,fc_layer_split, fc_layer_fixed_gates
from models.cl.continual_learner import ContinualLearner
from utils import get_data_loader
import functools
from itertools import chain
from models.attention import ExternalAttention

class AutoEncoder(ContinualLearner):
    """Class for variational auto-encoder (VAE) models."""

    def __init__(self, image_size, image_channels, classes,
                 # -conv-layers
                 conv_type="standard", depth=0, start_channels=64, reducing_layers=3, conv_bn=True, conv_nl="relu",
                 num_blocks=2, global_pooling=False, no_fnl=True, convE=None, conv_gated=False,
                 # -fc-layers
                 fc_layers=3, fc_units=1000, h_dim=400, fc_drop=0, fc_bn=False, fc_nl="relu", excit_buffer=False,
                 fc_gated=False,
                 # -prior
                 prior="standard", z_dim=20, per_class=False, n_modes=1,
                 # -decoder
                 recon_loss='BCE', network_output="sigmoid", deconv_type="standard", hidden=False,
                 dg_gates=False, dg_type="task", dg_prop=0., tasks=5, scenario="task", device='cuda',
                 # -classifer
                 classifier=True, classify_opt="beforeZ",
                 # -training-specific settings (can be changed after setting up model)
                 lamda_pl=0., lamda_rcl=1., lamda_vl=1., lamda_rep=1e-6, 
                 #### Determine whether or not to implement class repulsion...
                 repulsion=False, kl_js='js', use_rep_factor=False, rep_factor=20, apply_mask=False,
                 contrastive=False, c_temp=1.0, c_drop=0.5, contr_not_hidden=False, recon_repulsion=False, recon_rep_averaged=False,
                 lamda_recon_rep=1e-6, recon_attraction=False, lamda_recon_atr=1e-6, contr_scores=False, contr_hard=False,
                 simsiam=False, attention=False, ma=False, ma_drop=0.1, **kwargs):

        # Set configurations for setting up the model
        super().__init__()
        self.label = "VAE"
        self.image_size = image_size
        self.image_channels = image_channels
        self.classes = classes
        self.fc_layers = fc_layers
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.fc_units = fc_units
        self.fc_drop = fc_drop
        self.classify_opt = classify_opt
        self.depth = depth if convE is None else convE.depth
        # -replay hidden representations? (-> replay only propagates through fc-layers)
        self.hidden = hidden
        # -type of loss to be used for reconstruction
        self.recon_loss = recon_loss # options: BCE|MSE
        self.network_output = network_output
        # -settings for class- or task-specific gates in fully-connected hidden layers of decoder
        self.dg_type = dg_type
        self.dg_prop = dg_prop
        self.dg_gates = dg_gates if dg_prop>0. else False
        self.gate_size = (tasks if dg_type=="task" else classes) if self.dg_gates else 0
        self.scenario = scenario
        ####
        self.repulsion = repulsion
        self.recon_repulsion = recon_repulsion
        self.recon_rep_averaged = recon_rep_averaged
        self.recon_attraction = recon_attraction
        self.contr_not_hidden = contr_not_hidden
        self.contr_scores = contr_scores
        self.contr_hard = contr_hard
        ####
        # Optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []
        
        #### Encoder optimiser...
        self.E_optimizer = None
        self.E_optim_list = []

        # Prior-related parameters
        self.prior = prior
        self.per_class = per_class
        self.n_modes = n_modes*classes if self.per_class else n_modes
        self.modes_per_class = n_modes if self.per_class else None

        # Components deciding how to train / run the model (i.e., these can be changed after setting up the model)
        # -options for prediction loss
        self.lamda_pl = lamda_pl   # weight of classification-loss
        # -how to compute the loss function?
        self.lamda_rcl = lamda_rcl     # weight of reconstruction-loss
        self.lamda_vl = lamda_vl       # weight of variational loss
        ####
        self.lamda_rep = lamda_rep                 # weight of distribution repulsion loss
        self.lamda_recon_rep = lamda_recon_rep     # weight of recon repulsion loss
        self.lamda_recon_atr = lamda_recon_atr     # weight of recon attraction loss
        self.contrastive = contrastive
        self.c_temp = c_temp
        self.c_drop = c_drop
        ####
        ###lym
        self.simsiam = simsiam
        self.use_attention = attention
        self.ma = ma
        self.ma_drop = ma_drop

        # Check whether there is at least 1 fc-layer
        if fc_layers<1:
            raise ValueError("VAE cannot have 0 fully-connected layers!")

        ######------SPECIFY MODEL------######

        ##>----Encoder (= q[z|x])----<##
        self.convE = ConvLayers(conv_type=conv_type, block_type="basic", num_blocks=num_blocks,
                                image_channels=image_channels, depth=self.depth, start_channels=start_channels,
                                reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl,
                                output="none" if no_fnl else "normal", global_pooling=global_pooling,
                                gated=conv_gated) if (convE is None) else convE

        self.convE.to(self._device())
        self.flatten = modules.Flatten()
        #------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = self.convE.out_units(image_size)
        self.conv_out_size = self.convE.out_size(image_size)
        self.conv_out_channels = self.convE.out_channels

        if fc_layers<2:
            self.fc_layer_sizes = [self.conv_out_units]  #--> this results in self.fcE = modules.Identity()
        elif fc_layers==2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units]+[int(x) for x in np.linspace(fc_units, h_dim, num=fc_layers-1)]
        real_h_dim = h_dim if fc_layers>1 else self.conv_out_units
        #------------------------------------------------------------------------------------------#

        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                       excit_buffer=excit_buffer, gated=fc_gated)
        ###### Can add extra layers here ######
        ###lym attention
        print('Use Attention', self.use_attention)
        if self.use_attention:
            self.multihead_attn = torch.nn.MultiheadAttention(2000, 25, dropout=self.ma_drop, batch_first=True)
            self.multihead_attn.to(self._device())
            self.E_attn = ExternalAttention(2000, 25)
            self.E_attn.to(self._device())

        if self.contrastive:
            self.fcProj = MLP(size_per_layer=[2000, 2000], batch_norm=False, nl='relu',  # [2000, 100]
                              output='none')  # , final_norm=True)
            ###lym
            dim = 2000
            pred_dim = 512
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),  # hidden layer
                                           nn.Linear(pred_dim, dim))  # output layer

        # to z
        self.toZ = fc_layer_split(real_h_dim, z_dim, nl_mean='none', nl_logvar='none')#, drop=fc_drop)

        ##>----Classifier----<##
        if classifier:
            self.units_before_classifier = real_h_dim if self.classify_opt=='beforeZ' else z_dim
            self.classifier = fc_layer(self.units_before_classifier, classes, excit_buffer=True, nl='none')

        ##>----Decoder (= p[x|z])----<##
        out_nl = True if fc_layers > 1 else (True if (self.depth > 0 and not no_fnl) else False)
        real_h_dim_down = h_dim if fc_layers > 1 else self.convE.out_units(image_size, ignore_gp=True)
        if self.dg_gates:
            self.fromZ = fc_layer_fixed_gates(
                z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none",
                gate_size=self.gate_size, gating_prop=dg_prop, device=device
            )
        else:
            self.fromZ = fc_layer(z_dim, real_h_dim_down, batch_norm=(out_nl and fc_bn), nl=fc_nl if out_nl else "none")
        fc_layer_sizes_down = self.fc_layer_sizes
        fc_layer_sizes_down[0] = self.convE.out_units(image_size, ignore_gp=True)
        # -> if 'gp' is used in forward pass, size of first/final hidden layer differs between forward and backward pass
        if self.dg_gates:
            self.fcD = MLP_gates(
                size_per_layer=[x for x in reversed(fc_layer_sizes_down)], drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                gate_size=self.gate_size, gating_prop=dg_prop, device=device,
                output=self.network_output if (self.depth==0 or self.hidden) else 'normal',
            )
        else:
            self.fcD = MLP(
                size_per_layer=[x for x in reversed(fc_layer_sizes_down)], drop=fc_drop, batch_norm=fc_bn, nl=fc_nl,
                gated=fc_gated, output=self.network_output if (self.depth==0 or self.hidden) else 'normal',
            )
        # to image-shape
        self.to_image = modules.Reshape(image_channels=self.convE.out_channels if self.depth>0 else image_channels)
        # through deconv-layers
        self.convD = DeconvLayers(
            image_channels=image_channels, final_channels=start_channels, depth=self.depth,
            reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated,
            output=self.network_output, deconv_type=deconv_type,
        ) if (not self.hidden) else modules.Identity()

        if (self.contr_not_hidden or self.contrastive):
            self.convD_contr = DeconvLayers(
                image_channels=image_channels, final_channels=start_channels, depth=self.depth,
                reducing_layers=reducing_layers, batch_norm=conv_bn, nl=conv_nl, gated=conv_gated,
                output=self.network_output, deconv_type=deconv_type,
            )
        ##>----Prior----<##
        # -if using the GMM-prior, add its parameters
        if self.prior=="GMM":
            # -create
            self.z_class_means = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            self.z_class_logvars = nn.Parameter(torch.Tensor(self.n_modes, self.z_dim))
            # -initialize
            self.z_class_means.data.normal_()
            self.z_class_logvars.data.normal_()
        
        ### Whether to use JS-divergence instead of KL-divergence...
        self.kl_js = kl_js
        ### Whether to use the repulsion factor & its magnitude...
        self.use_rep_factor = use_rep_factor
        self.rep_factor = rep_factor
        self.apply_mask = apply_mask


    ##------ NAMES --------##

    def get_name(self):
        convE_label = "{}{}_".format(self.convE.name, "H" if self.hidden else "") if self.depth>0 else ""
        fcE_label = "{}_".format(self.fcE.name) if self.fc_layers>1 else "{}{}_".format("h" if self.depth>0 else "i",
                                                                                        self.conv_out_units)
        z_label = "z{}{}".format(self.z_dim, "" if self.prior=="standard" else "-{}{}{}".format(
            self.prior, self.n_modes, "pc" if self.per_class else ""
        ))
        class_label = "_c{}{}".format(
            self.classes, "" if self.classify_opt=="beforeZ" else self.classify_opt
        ) if hasattr(self, "classifier") else ""
        decoder_label = "_{}{}".format("tg" if self.dg_type=="task" else "cg", self.dg_prop) if self.dg_gates else ""
        return "{}={}{}{}{}{}".format(self.label, convE_label, fcE_label, z_label, class_label, decoder_label)

    @property
    def name(self):
        return self.get_name()


    ##------ LAYERS --------##

    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = []
        list += self.convE.list_init_layers()
        list += self.fcE.list_init_layers()
        list += self.fcProj.list_init_layers()
        if hasattr(self, "classifier"):
            list += self.classifier.list_init_layers()
        list += self.toZ.list_init_layers()
        list += self.fromZ.list_init_layers()
        list += self.fcD.list_init_layers()
        if not self.hidden:
            list += self.convD.list_init_layers()
        return list

    def layer_info(self):
        '''Return list with shape of all hidden layers.'''
        # create list with hidden convolutional layers
        layer_list = self.convE.layer_info(image_size=self.image_size) if not self.hidden else []
        # add output of final convolutional layer (if there was at least one conv-layer and there's fc-layers after)
        if (self.fc_layers>0 and self.depth>0) and not self.hidden:
            layer_list.append([self.conv_out_channels, self.conv_out_size, self.conv_out_size])
        # add layers of the MLP
        if self.fc_layers>1:
            for layer_id in range(1, self.fc_layers):
                layer_list.append([self.fc_layer_sizes[layer_id]])
        return layer_list


    ##------ FORWARD FUNCTIONS --------##

    def encode(self, x, not_hidden=False, use_views=False, batch_size=None, current=False):
        '''Pass input through feed-forward connections, to get [z_mean], [z_logvar] and [hE].
        Input [x] is either an image or, if [self.hidden], extracted "intermediate" or "internal" image features.'''
        # Forward-pass through conv-layers
        hidden_x = x if (self.hidden and not not_hidden) else self.convE(x)
        image_features = self.flatten(hidden_x)

        # Forward-pass through fc-layers
        #hE = self.fcE(image_features[:batch_size]) if self.contrastive else self.fcE(image_features)
        hE = self.fcE(image_features) #latent representation
        ######
        if self.contrastive and (not current):
            if self.use_attention:
                h_size = list(hE.size())
                hE_r = hE.reshape([1, h_size[0], h_size[1]])
                attn_hE = self.multihead_attn(hE_r, hE_r, hE_r)[0] if self.ma else self.E_attn(hE_r)
                attn_hE = attn_hE.reshape(h_size)
                # Drop-out random nodes...
                proj_z = F.normalize(self.fcProj(F.dropout(attn_hE, p=self.c_drop)), dim=1)
            else:
                # Drop-out random nodes...
                proj_z = F.normalize(self.fcProj(F.dropout(hE, p=self.c_drop)), dim=1)

            hE = hE[:batch_size]
        else:
            proj_z = None
        ######
        # Get parameters for reparametrization
        (z_mean, z_logvar) = self.toZ(hE)
        return z_mean, z_logvar, hE, hidden_x, proj_z

    def classify(self, x, not_hidden=False, reparameterize=True, current=False, **kwargs):
        '''For input [x] (image or extracted "internal" image features), return all predicted "scores"/"logits".'''
        if hasattr(self, "classifier"):
            hidden_x = x if (self.hidden and not not_hidden) else self.convE(x)
            # image_features = self.flatten(x) if (self.hidden and not not_hidden) else self.flatten(self.convE(x))
            image_features = self.flatten(hidden_x)
            hE = self.fcE(image_features)
            if self.classify_opt=="beforeZ":
                return self.classifier(hE)
            else:
                (mu, logvar) = self.toZ(hE)
                z = mu if (self.classify_opt=="fromZ" or (not reparameterize)) else self.reparameterize(mu, logvar)
                return self.classifier(z)
        else:
            return None

    def reparameterize(self, mu, logvar):
        '''Perform "reparametrization trick" to make these stochastic variables differentiable.'''
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()#.requires_grad_()
        return eps.mul(std).add_(mu)

    def decode(self, z, gate_input=None):
        '''Decode latent variable activations.

        INPUT:  - [z]            <2D-tensor>; latent variables to be decoded
                - [gate_input]   <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-/taskID  ---OR---
                                 <2D-tensor>; for each batch-element in [x] a probability for every class-/task-ID

        OUTPUT: - [image_recon]  <4D-tensor>'''

        # -if needed, convert [gate_input] to one-hot vector
        if self.dg_gates and (gate_input is not None) and (type(gate_input)==np.ndarray or gate_input.dim()<2):
            gate_input = lf.to_one_hot(gate_input, classes=self.gate_size, device=self._device())

        # -put inputs through decoder
        hD = self.fromZ(z, gate_input=gate_input) if self.dg_gates else self.fromZ(z)
        image_features = self.fcD(hD, gate_input=gate_input) if self.dg_gates else self.fcD(hD)
        image_recon = self.convD(self.to_image(image_features))
        return image_recon

    def forward(self, x, gate_input=None, full=False, reparameterize=True, use_views=False, batch_size=None, current=False, **kwargs):
        '''Forward function to propagate [x] through the encoder, reparametrization and decoder.

        Input: - [x]          <4D-tensor> of shape [batch_size]x[channels]x[image_size]x[image_size]
                              (or <4D-tensor> of shape [batch_size]x[out_channels]x[out_size]x[outsize], if self.hidden)
               - [gate_input] <1D-tensor> or <np.ndarray>; for each batch-element in [x] its class-ID (eg, [y]) ---OR---
                              <2D-tensor>; for each batch-element in [x] a probability for each class-ID (eg, [y_hat])

        If [full] is True, output should be a <tuple> consisting of:
        - [x_recon]     <4D-tensor> reconstructed image (features) in same shape as [x] (or 2 of those: mean & logvar)
        - [y_hat]       <2D-tensor> with predicted logits for each class
        - [mu]          <2D-tensor> with either [z] or the estimated mean of [z]
        - [logvar]      None or <2D-tensor> estimated log(SD^2) of [z]
        - [z]           <2D-tensor> reparameterized [z] used for reconstruction
        If [full] is False, output is simply the predicted logits (i.e., [y_hat]).'''
        if full:  ## Used for Class-IL...
            # -encode (forward), reparameterize and decode (backward)
            mu, logvar, hE, hidden_x, proj_z = self.encode(x, use_views=use_views, batch_size=batch_size, current=current)
            z = self.reparameterize(mu, logvar) if reparameterize else mu
            gate_input = gate_input if self.dg_gates else None
            x_recon = self.decode(z[:batch_size], gate_input=gate_input)
            # -classify
            if hasattr(self, "classifier"):
                if self.classify_opt in ["beforeZ", "fromZ"]:
                    y_hat = self.classifier(hE) if self.classify_opt=="beforeZ" else self.classifier(mu)
                else:
                    raise NotImplementedError("Classification-option {} not implemented.".format(self.classify_opt))
            else:
                y_hat = None
            # -return
            return (x_recon, y_hat, mu, logvar, z, proj_z)
        else:
            return self.classify(x, reparameterize=reparameterize, current=current) #-> if [full]=False, only forward pass for prediction

    def input_to_hidden(self, x):
        '''Get [hidden_rep]s (inputs to final fully-connected layers) for images [x].'''
        return self.convE(x)

    def feature_extractor(self, images, from_hidden=False):
        '''Extract "final features" (i.e., after both conv- and fc-layers of forward pass) from provided images.'''
        return self.fcE(self.flatten(images if from_hidden else self.convE(images)))


    ##------ SAMPLE FUNCTIONS --------##

    def sample(self, size, allowed_classes=None, class_probs=None, sample_mode=None, allowed_domains=None, specific_classes=None,
               only_x=False, only_z=False, **kwargs):
        '''Generate [size] samples from the model. Outputs are tensors (not "requiring grad"), on same device as <self>.

        INPUT:  - [allowed_classes]     <list> of [class_ids] from which to sample
                - [class_probs]         <list> with for each class the probability it is sampled from it
                - [sample_mode]         <int> to sample from specific mode of [z]-distr'n, overwrites [allowed_classes]
                - [allowed_domains]     <list> of [task_ids] which are allowed to be used for 'task-gates' (if used)
                                          NOTE: currently only relevant if [scenario]=="domain"
                - [specific_classes]    <tensor> of specific [class_ids] from which to sample, overwrites [sample_mode]

        OUTPUT: - [X]         <4D-tensor> generated images / image-features
                - [y_used]    <ndarray> labels of classes intended to be sampled  (using <class_ids>)
                - [task_used] <ndarray> labels of domains/tasks used for task-gates in decoder'''

        # set model to eval()-mode
        self.eval()

        # pick for each sample the prior-mode to be used
        if self.prior=="GMM":
            if specific_classes is None:
                if sample_mode is None:
                    if (allowed_classes is None and class_probs is None) or (not self.per_class):
                        # -randomly sample modes from all possible modes (and find their corresponding class, if applicable)
                        sampled_modes = np.random.randint(0, self.n_modes, size)
                        y_used = np.array(
                            [int(mode / self.modes_per_class) for mode in sampled_modes]
                        ) if self.per_class else None
                    else:
                        if allowed_classes is None:
                            allowed_classes = [i for i in range(len(class_probs))]
                        # -sample from modes belonging to [allowed_classes], possibly weighted according to [class_probs]
                        allowed_modes = []     # -collect all allowed modes
                        unweighted_probs = []  # -collect unweighted sample-probabilities of those modes
                        for index, class_id in enumerate(allowed_classes):
                            allowed_modes += list(range(class_id * self.modes_per_class, (class_id+1)*self.modes_per_class))
                            if class_probs is not None:
                                for i in range(self.modes_per_class):
                                    unweighted_probs.append(class_probs[index].item())
                        mode_probs = None if class_probs is None else [p / sum(unweighted_probs) for p in unweighted_probs]
                        sampled_modes = np.random.choice(allowed_modes, size, p=mode_probs, replace=True)
                        y_used = np.array([int(mode / self.modes_per_class) for mode in sampled_modes])
                            
                else:
                    # -always sample from the provided mode
                    sampled_modes = np.repeat(sample_mode, size)
                    y_used = np.repeat(int(sample_mode / self.modes_per_class), size) if self.per_class else None

            else: #### Getting random modes from specific list of classes...
                sampled_modes = specific_classes
                ####
                
        else:
            y_used = None

        # sample z
        if self.prior=="GMM":
            #### Return only the mean & logvar of the z-distribution for the specific classes...
            if (specific_classes is not None) and (only_z == True):
                prior_means = self.z_class_means
                prior_logvars = self.z_class_logvars
                # -for each sample to be generated, select the previously sampled mode
                z_means = prior_means[sampled_modes]
                z_logvars = prior_logvars[sampled_modes]
                # set model to train()-mode
                self.train()
                return z_means, z_logvars
            ####
            else:
                prior_means = self.z_class_means
                prior_logvars = self.z_class_logvars
                # -for each sample to be generated, select the previously sampled mode
                z_means = prior_means[sampled_modes, :]
                z_logvars = prior_logvars[sampled_modes, :]

            with torch.no_grad():
                z = self.reparameterize(z_means, z_logvars)

        else:
            z = torch.randn(size, self.z_dim).to(self._device())

        # if no classes are selected yet, but they are needed for the "decoder-gates", select classes to be sampled
        if (y_used is None) and (self.dg_gates):
            if allowed_classes is None and class_probs is None:
                y_used = np.random.randint(0, self.classes, size)
            else:
                if allowed_classes is None:
                    allowed_classes = [i for i in range(len(class_probs))]
                y_used = np.random.choice(allowed_classes, size, p=class_probs, replace=True)
        # if the gates in the decoder are "task-gates", convert [y_used] to corresponding tasks (if Task-IL or Class-IL)
        #   or simply sample which tasks should be generated (if Domain-IL) from [allowed_domains]
        task_used = None
        if self.dg_gates and self.dg_type=="task":
            if self.scenario=="domain":
                task_used = np.random.randint(0,self.gate_size,size) if (allowed_domains is None) else np.random.choice(
                    allowed_domains, size, replace=True
                )
            else:
                classes_per_task = int(self.classes/self.gate_size)
                task_used = np.array([int(class_id / classes_per_task) for class_id in y_used])

        # decode z into image X
        with torch.no_grad():
            X = self.decode(z, gate_input=(task_used if self.dg_type=="task" else y_used) if self.dg_gates else None)

        # return samples as [batch_size]x[channels]x[image_size]x[image_size] tensor, plus requested additional info
        if only_x:
            return X
        elif self.contr_not_hidden or self.contrastive:
            X_imgs = self.convD_contr(X)
            return (X, y_used, task_used, X_imgs)
        else:
            return (X, y_used, task_used)



    ##------ LOSS FUNCTIONS --------##

    def calculate_recon_loss(self, x, x_recon, average=False):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [x]           <tensor> with original input (1st dimension (ie, dim=0) is "batch-dimension")
                - [x_recon]     (tuple of 2x) <tensor> with reconstructed input in same shape as [x]
                - [average]     <bool>, if True, loss is average over all pixels; otherwise it is summed

        OUTPUT: - [reconL]      <1D-tensor> of length [batch_size]'''

        batch_size = x.size(0)
        if self.recon_loss=="MSE":
            # reconL = F.mse_loss(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1), reduction='none')
            # reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
            reconL = -lf.log_Normal_standard(x=x, mean=x_recon, average=average, dim=-1)
        elif self.recon_loss=="BCE":
            reconL = F.binary_cross_entropy(input=x_recon.view(batch_size, -1), target=x.view(batch_size, -1),
                                            reduction='none')
            reconL = torch.mean(reconL, dim=1) if average else torch.sum(reconL, dim=1)
        else:
            raise NotImplementedError("Wrong choice for type of reconstruction-loss!")
        # --> if [average]=True, reconstruction loss is averaged over all pixels/elements (otherwise it is summed)
        #       (averaging over all elements in the batch will be done later)
        return reconL


    def calculate_log_p_z(self, z, y=None, y_prob=None, allowed_classes=None):
        '''Calculate log-likelihood of sampled [z] under the prior distirbution.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [log_p_z]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            log_p_z = lf.log_Normal_standard(z, average=False, dim=1)   # [batch_size]

        if self.prior == "GMM":
            ## Get [means] and [logvars] of all (possible) modes
            allowed_modes = list(range(self.n_modes))
            # -if we don't use the specific modes of a target, we could select modes based on list of classes
            if (y is None) and (allowed_classes is not None) and self.per_class:
                allowed_modes = []
                for class_id in allowed_classes:
                    allowed_modes += list(range(class_id * self.modes_per_class, (class_id + 1) * self.modes_per_class))
            # -calculate/retireve the means and logvars for the selected modes
            prior_means = self.z_class_means[allowed_modes, :]
            prior_logvars = self.z_class_logvars[allowed_modes, :]
            # -rearrange / select for each batch prior-modes to be used
            z_expand = z.unsqueeze(1)  # [batch_size] x 1 x [z_dim]
            means = prior_means.unsqueeze(0)  # 1 x [n_modes] x [z_dim]
            logvars = prior_logvars.unsqueeze(0)  # 1 x [n_modes] x [z_dim]

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            n_modes = self.modes_per_class if (
                ((y is not None) or (y_prob is not None)) and self.per_class
            ) else len(allowed_modes)
            a = lf.log_Normal_diag(z_expand, mean=means, log_var=logvars, average=False, dim=2) - math.log(n_modes)
            # --> for each element in batch, calculate log-likelihood for all pseudoinputs: [batch_size] x [n_modes]
            if (y is not None) and self.per_class:
                modes_list = list()
                for i in range(len(y)):
                    target = y[i].item()
                    modes_list.append(list(range(target * self.modes_per_class, (target + 1) * self.modes_per_class)))
                modes_tensor = torch.LongTensor(modes_list).to(self._device())
                a = a.gather(dim=1, index=modes_tensor)
                # --> reduce [a] to size [batch_size]x[modes_per_class] (ie, per batch only keep modes of [y])
                #     but within the batch, elements can have different [y], so this reduction couldn't be done before
            a_max, _ = torch.max(a, dim=1)  # [batch_size]
            # --> for each element in batch, take highest log-likelihood over all pseudoinputs
            #     this is calculated and used to avoid underflow in the below computation
            a_exp = torch.exp(a - a_max.unsqueeze(1))  # [batch_size] x [n_modes]
            if (y is None) and (y_prob is not None) and self.per_class:
                batch_size = y_prob.size(0)
                y_prob = y_prob.view(-1, 1).repeat(1, self.modes_per_class).view(batch_size, -1)
                # ----> extend probabilities per class to probabilities per mode; y_prob: [batch_size] x [n_modes]
                a_logsum = torch.log(torch.clamp(torch.sum(y_prob * a_exp, dim=1), min=1e-40))
            else:
                a_logsum = torch.log(torch.clamp(torch.sum(a_exp, dim=1), min=1e-40))  # -> sum over modes: [batch_size]
            log_p_z = a_logsum + a_max  # [batch_size]
            
        return log_p_z


    def calculate_variat_loss(self, z, mu, logvar, y=None, y_prob=None, allowed_classes=None):
        '''Calculate reconstruction loss for each element in the batch.

        INPUT:  - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")
                - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OPTIONS THAT ARE RELEVANT ONLY IF self.per_class IS TRUE:
            - [y]               None or <1D-tensor> with target-classes (as integers)
            - [y_prob]          None or <2D-tensor> with probabilities for each class (in [allowed_classes])
            - [allowed_classes] None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [variatL]   <1D-tensor> of length [batch_size]'''

        if self.prior == "standard":
            # --> calculate analytically
            # ---- see Appendix B from: Kingma & Welling (2014) Auto-Encoding Variational Bayes, ICLR ----#
            variatL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        elif self.prior=="GMM":
            # --> calculate "by estimation"

            ## Calculate "log_p_z" (log-likelihood of "reparameterized" [z] based on selected priors)
            log_p_z = self.calculate_log_p_z(z, y=y, y_prob=y_prob, allowed_classes=allowed_classes)
            # ----->  log_p_z: [batch_size]

            ## Calculate "log_q_z_x" (entropy of "reparameterized" [z] given [x])
            log_q_z_x = lf.log_Normal_diag(z, mean=mu, log_var=logvar, average=False, dim=1)
            # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
            # ----->  log_q_z_x: [batch_size]

            ## Combine
            variatL = -(log_p_z - log_q_z_x)

        return variatL
    
    #### New losses added here ####

    def calculate_diff_loss(self, mu_1, logvar_1, mu_2, logvar_2, kl_js='js', keep_inds=None, similarity=None, attract=False):
        '''Calculate distribution repulsion loss for each element in the batch.

        INPUT:  - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]
                - [z]        <2D-tensor> with sampled latent variables (1st dimension (ie, dim=0) is "batch-dimension")

        OUTPUT: - [diffL]   <1D-tensor> of length [batch_size]'''
        
        #### KL-divergence between two gaussian distributions...
        # --> calculate analytically
        if (keep_inds is not None) and (len(keep_inds)!=list(mu_1.shape)[0]):
            mu_1, logvar_1, mu_2, logvar_2 = mu_1[keep_inds], logvar_1[keep_inds], mu_2[keep_inds], logvar_2[keep_inds]

        if kl_js=='js':
            ## JS-divergence...
            mu_m, logvar_m = 0.5 * (mu_1 + mu_2), torch.log(0.25 * (torch.exp(logvar_1) + torch.exp(logvar_2)))
    
            diffL_1 = 0.5 * torch.sum(torch.exp(logvar_1 - logvar_m) + torch.mul(torch.pow((mu_m - mu_1), 2), torch.exp(-logvar_m)) + logvar_m - logvar_1 - 1, dim=1)
            diffL_2 = 0.5 * torch.sum(torch.exp(logvar_2 - logvar_m) + torch.mul(torch.pow((mu_m - mu_2), 2), torch.exp(-logvar_m)) + logvar_m - logvar_2 - 1, dim=1)
            
            diffL = 0.5 * (diffL_1 + diffL_2)
        else:
            diffL = 0.5 * torch.sum(torch.exp(logvar_1 - logvar_2) + torch.mul(torch.pow((mu_2 - mu_1), 2), torch.exp(-logvar_2)) + logvar_2 - logvar_1 - 1, dim=1)
        if similarity is None:
            if attract:
                return diffL
            else:
                # Taking the inverse of the divergence...
                return torch.pow(diffL, -1)
        else:
            diffL = torch.pow(diffL, -1)
            return diffL
    
    def calculate_rep2_loss(self, z_1, mu_1, logvar_1, z_2, mu_2, logvar_2):
        '''Calculate difference loss for each element in the batch.

        INPUT:  - [mu]       <2D-tensor> by encoder predicted mean for [z]
                - [logvar]   <2D-tensor> by encoder predicted logvar for [z]

        OUTPUT: - [diffL]   <1D-tensor> of length [batch_size]'''
        
        # --> calculate "by estimation"

        ## Calculate "log_q1_z_x" (entropy of "reparameterized" [z] given [x])
        log_q1_z_x = lf.log_Normal_diag(z_1, mean=mu_1, log_var=logvar_1, average=False, dim=1)
        # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
        # ----->  log_q_z_x: [batch_size]

        ## Calculate "log_q2_z_x" (entropy of "reparameterized" [z] given [x])
        log_q2_z_x = lf.log_Normal_diag(z_2, mean=mu_2, log_var=logvar_2, average=False, dim=1)
        # ----->  mu: [batch_size] x [z_dim]; logvar: [batch_size] x [z_dim]; z: [batch_size] x [z_dim]
        # ----->  log_q_z_x: [batch_size]

        ## Combine
        diffL = torch.exp(log_q1_z_x) * (log_q1_z_x - log_q2_z_x) + 1e-6
        diffL = torch.pow(diffL, -1)
        return diffL

    def calculate_contr_loss(self, proj_z, y, scores=None, base_temp=0.07):
        '''Calculate contrastive loss on encoder and projection head.
        
        INPUT:  - [proj_z]     <3D tensor> [batch_size]x[n_views]x[proj_output_size]

        OUTPUT: - [contrL]     <1D tensor> of length [batch_size]'''
        
        temp = self.c_temp
        use_scores = self.contr_scores
        hard_sampling = self.contr_hard
        y = scores if use_scores and (scores is not None) else y
        
        batch_size = proj_z.shape[0]
        y = y.contiguous() if use_scores and (scores is not None) else y.contiguous().view(-1, 1)
        if y.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features!!')

        mask = torch.matmul(y, y.T).to(self._device()) if use_scores and (scores is not None) else torch.eq(y, y.T).float().to(self._device())
        
        contr_count = proj_z.shape[1]
        contr_feature = torch.cat(torch.unbind(proj_z, dim=1), dim=0)
        anchor_feature = contr_feature
        anchor_count = contr_count

        anchor_dot_contr = torch.div(
            torch.matmul(anchor_feature, contr_feature.T), temp)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contr, dim=1, keepdim=True)
        logits = anchor_dot_contr - logits_max.detach()  

        # Tile mask
        mask = mask.repeat(anchor_count, contr_count)

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self._device()), 0)

        mask = mask * logits_mask
        
        ## Hard sampling (a)...
        if hard_sampling:
            tau = 0.05
            beta = 1.0
            N_neg = ((list(y.shape)[0]*2) - mask.sum(1)).detach()
            neg_mask = logits_mask - mask
            pos_mask = mask
            exp_logits = torch.exp(logits)

            exp_neg = exp_logits * neg_mask
            exp_pos = exp_logits * pos_mask

            exp_neg_beta = torch.pow(exp_neg, beta)
            reweighted_neg = exp_neg_beta.sum(1) / (exp_neg_beta.mean(1))            

            Ng_sum = ((- N_neg * tau * exp_pos) + (reweighted_neg * exp_neg)).sum(1, keepdim=True) / (1 - tau)
            Ng_sum = torch.where(Ng_sum < (N_neg * torch.exp(torch.tensor(-1 / temp))), N_neg * torch.exp(torch.tensor(-1 / temp)), Ng_sum)
            pos_sum = exp_pos.sum(1, keepdim=True)

        else:
            exp_logits = torch.exp(logits)

        # Compute log_prob
        exp_logits = exp_logits * logits_mask
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) if not hard_sampling else logits - torch.log(Ng_sum + pos_sum)

        # Compute mean of log-likelihood over positive: sum[log(exp/sum(exp))] / |P(i)|
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Contrastive loss...
        contrL = - (temp / base_temp) * mean_log_prob_pos
        return contrL.view(anchor_count, batch_size).mean()


    #### End of new losses ####

    #### Main loss function below has been heavily modified to include the new losses #### 

    def loss_function(self, x, y, x_recon, y_hat, scores, mu, z, logvar=None, allowed_classes=None, batch_weights=None,
                      diff=False, mu_diff=None, logvar_diff=None, mu_2=None, logvar_2=None, mu_3=None, logvar_3=None,
                      mu_4=None, logvar_4=None, kl_js='js', use_rep_factor=False, mu_b=None, logvar_b=None,
                      mu_b_sim=None, logvar_b_sim=None, keep_inds=None, similarity=None, proj_z=None, use_views=False, x_rep=None,
                      x_recon_rep=None, x_atr=None, x_recon_atr=None):
        '''Calculate and return various losses that could be used for training and/or evaluating the model.

        INPUT:  - [x]           <4D-tensor> original image
                - [y]           <1D-tensor> with target-classes (as integers, corresponding to [allowed_classes])
                - [x_recon]     (tuple of 2x) <4D-tensor> reconstructed image in same shape as [x]
                - [y_hat]       <2D-tensor> with predicted "logits" for each class (corresponding to [allowed_classes])
                - [scores]         <2D-tensor> with target "logits" for each class (corresponding to [allowed_classes])
                                     (if len(scores)<len(y_hat), 0 probs are added during distillation step at the end)
                - [mu]             <2D-tensor> with either [z] or the estimated mean of [z]
                - [z]              <2D-tensor> with reparameterized [z]
                - [logvar]         None or <2D-tensor> with estimated log(SD^2) of [z]
                - [batch_weights]  <1D-tensor> with a weight for each batch-element (if None, normal average over batch)
                - [allowed_classes]None or <list> with class-IDs to use for selecting prior-mode(s)

        OUTPUT: - [reconL]       reconstruction loss indicating how well [x] and [x_recon] match
                - [variatL]      variational (KL-divergence) loss "indicating how close distribion [z] is to prior"
                - [predL]        prediction loss indicating how well targets [y] are predicted
                - [distilL]      knowledge distillation (KD) loss indicating how well the predicted "logits" ([y_hat])
                                     match the target "logits" ([scores])'''
            
        ###-----Reconstruction loss-----###
        # Calculate original reconstruction loss...
        batch_size = x.size(0)
        reconL = self.calculate_recon_loss(x=x.view(batch_size, -1), average=True,
                                           x_recon=x_recon.view(batch_size, -1)) # -> average over pixels
        reconL = lf.weighted_average(reconL, weights=batch_weights, dim=0)       # -> average over batch
        
        # Calculate reconstruction repulsion loss...
        if self.recon_repulsion and (x_rep is not None) and (len(keep_inds) > 0):
            # Remove samples for which examples of the competing class could not be found...
            bsz = list(x_recon_rep.shape)[0]
            if (keep_inds is not None) and (len(keep_inds)!=list(x_rep.view(bsz, -1).shape)[0]):
                x_rep, x_recon_rep = x_rep.view(bsz, -1)[keep_inds], x_recon_rep.view(bsz, -1)[keep_inds]
            else:
                x_rep, x_recon_rep = x_rep.view(bsz, -1), x_recon_rep.view(bsz, -1)

            # Recon repulsion loss...
            recon_repL = torch.pow(self.calculate_recon_loss(x=x_rep, average=True,
                                           x_recon=x_recon_rep), -1)
            # Average over the batch...
            recon_repL = lf.weighted_average(recon_repL, weights=batch_weights, dim=0)
        else:
            recon_repL = None
        
        # Calculate reconstruction attraction loss...
        if self.recon_attraction and (x_atr is not None):
            recon_atrL = self.calculate_recon_loss(x=x_atr.view(batch_size, -1), average=True,
                                           x_recon=x_recon_atr.view(batch_size, -1))
            recon_atrL = lf.weighted_average(recon_atrL, weights=batch_weights, dim=0)
        else:
            recon_atrL = None

        ###-----Variational loss-----###
        if logvar is not None:
            actual_y = torch.tensor([allowed_classes[i.item()] for i in y]).to(self._device()) if (
                (allowed_classes is not None) and (y is not None)
            ) else y
            if (y is None and scores is not None):
                y_prob = F.softmax(scores / self.KD_temp, dim=1)
                if allowed_classes is not None and len(allowed_classes) > y_prob.size(1):
                    n_batch = y_prob.size(0)
                    zeros_to_add = torch.zeros(n_batch, len(allowed_classes) - y_prob.size(1))
                    zeros_to_add = zeros_to_add.to(self._device())
                    y_prob = torch.cat([y_prob, zeros_to_add], dim=1)
            else:
                y_prob = None
            # ---> if [y] is not provided but [scores] is, calculate variational loss using weighted sum of prior-modes
            variatL = self.calculate_variat_loss(z=z, mu=mu, logvar=logvar, y=actual_y, y_prob=y_prob,
                                                 allowed_classes=allowed_classes)
            variatL = lf.weighted_average(variatL, weights=batch_weights, dim=0)  # -> average over batch
            variatL /= (self.image_channels * self.image_size ** 2)               # -> divide by # of input-pixels
        else:
            variatL = torch.tensor(0., device=self._device())
        
        ###-----Difference loss-----###
        
        diff = False ##########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#########################################
        
        #### Difference loss for each image in batch...
        if diff:
            if (mu_2 is not None) and (logvar_2 is not None):
                diffL = self.calculate_diff_loss(mu_1=mu if mu_diff is None else mu_diff, logvar_1=logvar if logvar_diff is None else logvar_diff, \
                                                 mu_2=mu_2, logvar_2=logvar_2, kl_js=kl_js)
                diffL = lf.weighted_average(diffL, weights=batch_weights, dim=0)
                diffL /= (self.image_channels * self.image_size ** 2)
                
            else:
                diffL = None
    
            if (mu_3 is not None) and (logvar_3 is not None):
                diffL_2 = self.calculate_diff_loss(mu_1=mu_3[0] if use_rep_factor else mu, logvar_1=logvar_3[0] if use_rep_factor else logvar, mu_2=mu_3[1] if use_rep_factor else mu_3, \
                                                   logvar_2=logvar_3[1] if use_rep_factor else logvar_3, kl_js=kl_js)
                diffL_2 = lf.weighted_average(diffL_2, weights=batch_weights, dim=0)
                diffL_2 /= (self.image_channels * self.image_size ** 2)
            else:
                diffL_2 = None
    
            if (mu_4 is not None) and (logvar_4 is not None):
                diffL_3 = self.calculate_diff_loss(mu_1=mu_4[0] if use_rep_factor else mu, logvar_1=logvar_4[0] if use_rep_factor else logvar, mu_2=mu_4[1] if use_rep_factor else mu_4, \
                                                   logvar_2=logvar_4[1] if use_rep_factor else logvar_4, kl_js=kl_js)
                diffL_3 = lf.weighted_average(diffL_3, weights=batch_weights, dim=0)
                diffL_3 /= (self.image_channels * self.image_size ** 2)
            else:
                diffL_3 = None
        else:
            diffL, diffL_2, diffL_3 = None, None, None

        rep2 = True
        if rep2:
            if (mu_b is not None) and (logvar_b is not None):
                diffL = self.calculate_overlap_loss(mu_1=mu, logvar_1=logvar, mu_2=mu_b, logvar_2=logvar_b)
                diffL = lf.weighted_average(diffL, weights=batch_weights, dim=0)
                diffL_2 = None
            else:
                diffL, diffL_2 = None, None
        ####
        
        ###-----Prediction loss-----###
        if y is not None and y_hat is not None:
            predL = F.cross_entropy(input=y_hat, target=y, reduction='none')
            #--> no reduction needed, summing over classes is "implicit"
            predL = lf.weighted_average(predL, weights=batch_weights, dim=0)  # -> average over batch
        else:
            predL = torch.tensor(0., device=self._device())

        ###-----Distilliation loss-----###
        if scores is not None and y_hat is not None:
            # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes would be added to [scores]!
            n_classes_to_consider = y_hat.size(1)    #--> zeros will be added to [scores] to make it this size!
            distilL = lf.loss_fn_kd(scores=y_hat[:, :n_classes_to_consider], target_scores=scores, T=self.KD_temp,
                                    weights=batch_weights)  #--> summing over classes & averaging over batch in function
        else:
            distilL = torch.tensor(0., device=self._device())

        if (proj_z is not None) and (self.contrastive):
            y = torch.argmax(scores, dim=1) if y is None else y
            contrL = self.calculate_contr_loss(proj_z, y, scores)
        else:
            contrL = None

        # Return a tuple of the calculated losses
        if diffL is None:
            if (not self.contrastive) or (proj_z is None):
                return reconL, variatL, predL, distilL, contrL, recon_repL, recon_atrL
            else:
                return reconL, variatL, predL, distilL, contrL, recon_repL, recon_atrL
        elif diffL_2 is None:
            return reconL, variatL, diffL, predL, distilL, contrL, recon_repL, recon_atrL
        elif diffL_3 is None:
            return reconL, variatL, diffL, diffL_2, predL, distilL, contrL, recon_repL, recon_atrL
        else:
            return reconL, variatL, diffL, diffL_2, diffL_3, predL, distilL, contrL, recon_repL, recon_atrL

    #### End of main loss function modifications ####

    ##------ EVALUATION FUNCTIONS --------##

    def calculate_recon_error(self, dataset, batch_size=128, max_repatches=None, average=False):
        '''Calculate reconstruction error of the model for each datapoint in [dataset].

        [average]     <bool>, if True, reconstruction-error is averaged over all pixels/units; otherwise it is summed'''

        # This function currently does not (always) work for Task-IL scenario or for decoder-gates with [dg_type]="task"
        if self.scenario=="task" or (self.dg_gates and self.dg_prop>0. and self.dg_type=="task"):
            raise NotImplementedError(
                "Function 'calculate_recon_error' not yet implemented for Task-IL scenario or task-based decoder-gates"
            )

        # Create data-loader
        data_loader = get_data_loader(dataset, batch_size=batch_size, cuda=self._is_on_cuda())

        # Break loop if max number of batches has been reached
        for index, (x, y) in enumerate(data_loader):
            if max_repatches is not None and index >= max_repatches:
                break

            # Move [x] and [y] to correct device
            x = x.to(self._device())
            y = y.to(self._device())

            # If internal replay, convert inputs to hidden feature representations
            if self.hidden:
                with torch.no_grad():
                    x = self.input_to_hidden(x)

            # Run forward pass of model to get [z_mean]
            with torch.no_grad():
                z_mean, _, _, _, _ = self.encode(x)

            # Run backward pass of model to reconstruct input
            gate_input = y.expand(x.size(0)) if self.dg_gates else None
            with torch.no_grad():
                x_recon = self.decode(z_mean, gate_input=gate_input)

            # Calculate reconstruction error
            recon_error = self.calculate_recon_loss(x.view(x.size(0), -1), x_recon.view(x.size(0), -1), average=average)

            # Concatanate the calculated reconstruction errors for all evaluated samples
            all_res = torch.cat([all_res, recon_error]) if index > 0 else recon_error

        # Convert to <np-array> (with one entry for each evaluated sample in [dataset]) and return
        return all_res.cpu().numpy()


    def estimate_loglikelihood(self, dataset, S=5000, batch_size=128, max_n=None):
        '''Estimate average marginal log-likelihood for x|y of the model on [dataset] using [S] importance samples.'''

        # This function currently does not (always) work for Task-IL scenario or for decoder-gates with [dg_type]="task"
        if self.scenario=="task" or (self.dg_gates and self.dg_prop>0. and self.dg_type=="task"):
            raise NotImplementedError(
                "Function 'estimate_loglikelihood' not yet implemented for Task-IL scenario or task-based decoder-gates"
            )

        # Create data-loader to give batches of size 1
        data_loader = get_data_loader(dataset, batch_size=1, cuda=self._is_on_cuda())

        # List to store estimated log-likelihood for each datapoint
        ll_per_datapoint = []

        # Break loop if max number of samples has been reached
        for index, (x, y) in enumerate(data_loader):
            if max_n is not None and index >= max_n:
                break

            # Move [x] and [y] to correct device
            x = x.to(self._device())
            y = y.to(self._device())

            # If hidden replay, convert inputs to hidden feature representations
            if self.hidden:
                with torch.no_grad():
                    x = self.input_to_hidden(x)

            # Run forward pass of model to get [z_mu] and [z_logvar]
            with torch.no_grad():
                z_mu, z_logvar, _, _, _ = self.encode(x)

            # Importance samples will be calcualted in batches, get number of required batches
            repeats = int(np.ceil(S / batch_size))

            # For each importance sample, calculate log_likelihood
            for rep in range(repeats):
                batch_size_current = (S % batch_size) if rep==(repeats-1) else batch_size

                # Reparameterize (i.e., sample z_s)
                z = self.reparameterize(z_mu.expand(batch_size_current, -1), z_logvar.expand(batch_size_current, -1))

                # Calculate log_p_z
                with torch.no_grad():
                    log_p_z = self.calculate_log_p_z(z, y=y.expand(batch_size_current))

                # Calculate log_q_z_x
                log_q_z_x = lf.log_Normal_diag(z, mean=z_mu, log_var=z_logvar, average=False, dim=1)

                # Calcuate p_x_z
                # -reconstruct input
                gate_input = y.expand(batch_size_current) if self.dg_gates else None
                with torch.no_grad():
                    x_recon = self.decode(z, gate_input=gate_input)
                # -calculate p_x_z (under Gaussian observation model with unit variance)
                log_p_x_z = lf.log_Normal_standard(x=x, mean=x_recon, average=False, dim=-1)

                # Calculate log-likelihood for each importance sample
                log_likelihoods = log_p_x_z + log_p_z - log_q_z_x

                # Concatanate the log-likelihoods of all importance samples
                all_lls = torch.cat([all_lls, log_likelihoods]) if rep > 0 else log_likelihoods

            # Calculate average log-likelihood over all importance samples for this test sample
            #  (for this, convert log-likelihoods back to likelihoods before summing them!)
            log_likelihood = all_lls.logsumexp(dim=0) - np.log(S)

            # Add it to list
            ll_per_datapoint.append(log_likelihood.cpu().numpy())

        return ll_per_datapoint



    ##------ TRAINING FUNCTIONS --------##

    def train_a_batch(self, x, y=None, x_=None, y_=None, scores_=None, top_scores_=None, top_threshold=None,
                      batch_index=None, tasks_=None, rnt=0.5,
                      active_classes=None, task=1, replay_not_hidden=False, freeze_convE=False, batch_size=None, 
                      batch_size_replay=None, task_n=None, use_views=False, 
                      contrast_current=False, contrast_replayed=True, criterion=None, **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 None or <tensor> batch of corresponding labels
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
                              NOTE: expected to be at hidden level if [self.hidden], unless [replay_not_hidden]==True
        [y_]                None or (<list> of) <1Dtensor>:[batch] of corresponding "replayed" labels
        [scores_]           None or (<list> of) <2Dtensor>:[batch]x[classes] target "scores"/"logits" for [x_]
        [tasks_]            None or (<list> of) <1Dtensor>/<ndarray>:[batch] of task-IDs of replayed samples (as <int>)
        [rnt]               <number> in [0,1], relative importance of new task
        [active_classes]    None or (<list> of) <list> with "active" classes
        [task]              <int>, for setting task-specific mask
        [replay_not_hidden] <bool> provided [x_] are original images, even though other level might be expected'''
        
        if self.contrastive:
            if contrast_replayed:
                x_ = torch.cat([x_[0], x_[1]], dim=0) if x_ is not None else None
            if contrast_current:
                x = torch.cat([x[0], x[1]], dim=0) if x is not None else None
            ###0821
            for param in chain(self.fcProj.parameters(), self.predictor.parameters()):
                param.requires_grad = True

        # Set model to training-mode
        self.train()
        if (freeze_convE) and (not (self.contrastive and contrast_current)):
            # - if conv-layers are frozen, they shoud be set to eval() to prevent batch-norm layers from changing
            self.convE.eval()

        # Reset optimizer
        self.optimizer.zero_grad()

        #### Reset encoder optimizer...
        if self.contrastive:
            self.E_optimizer.zero_grad()

        ##--(1)-- CURRENT DATA --##
        precision = 0.
        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # If using task-gates, create [task_tensor] as it's needed in the decoder
            task_tensor = None
            if self.dg_gates and self.dg_type=="task":
                task_tensor = torch.tensor(np.repeat(task-1, x.size(0))).to(self._device())

            # Run the model
            x = self.convE(x) if self.hidden else x   # -pre-processing (if 'hidden')
            recon_batch, y_hat, mu, logvar, z, proj_z = self(
                x, gate_input=(task_tensor if self.dg_type=="task" else y) if self.dg_gates else None, full=True,
                reparameterize=True, use_views=use_views, batch_size=batch_size, current=True
                )
            if self.contrastive and contrast_current:
                proj_z1, proj_z2 = torch.split(proj_z, [batch_size, batch_size], dim=0)
                proj_z = torch.cat([proj_z1.unsqueeze(1), proj_z2.unsqueeze(1)], dim=1)
                p1 = self.predictor(proj_z1)
                p2 = self.predictor(proj_z2)
                proj_z1.detach()  # stop gradient
                proj_z2.detach()
                x = x[:batch_size]
                ###lym
                ss_loss = -(criterion(p1, proj_z2).mean() + criterion(p2, proj_z1).mean()) * 0.5

            # -if needed ("class"/"task"-scenario), find allowed classes for current task & remove predictions of others
            if active_classes is not None:
                class_entries = active_classes[-1] if type(active_classes[0])==list else active_classes
                if y_hat is not None:
                    y_hat = y_hat[:, class_entries]

            # Calculate all losses ###
            if (not self.contrastive) or (not contrast_current):
                reconL, variatL, predL, _, _, _, _ = self.loss_function(
                    x=x, y=y, x_recon=recon_batch, y_hat=y_hat, scores=None, mu=mu, z=z, logvar=logvar,
                    allowed_classes=class_entries if active_classes is not None else None, use_views=use_views)
                #--> [allowed_classes] will be used only if [y] is not provided
            else:
                reconL, variatL, predL, _, contrL, _, _ = self.loss_function(
                    x=x, y=y, x_recon=recon_batch, y_hat=y_hat, scores=None, mu=mu, z=z, logvar=logvar,
                    allowed_classes=class_entries if active_classes is not None else None, proj_z=proj_z, use_views=use_views)

            # Weigh losses as requested
            loss_cur = self.lamda_rcl*reconL + self.lamda_vl*variatL + self.lamda_pl*predL

            # Calculate training-precision
            if y is not None and y_hat is not None:
                _, predicted = y_hat.max(1)
                precision = (y == predicted).sum().item() / x.size(0)

            # If XdG is combined with replay, backward-pass needs to be done before new task-mask is applied
            if (self.mask_dict is not None) and (x_ is not None):
                weighted_current_loss = rnt*loss_cur
                #### Before optimisation step, set requires_grad = False for encoder &
                #### projection head...
                if self.contrastive:
                    for param in self.parameters():
                        param.requires_grad = True
                    if not self.use_attention:
                        for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters()):
                            param.requires_grad = False
                    else:
                        for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters(),
                                           self.multihead_attn.parameters(), self.E_attn.parameters()):
                            param.requires_grad = False

                # Update gradients...
                weighted_current_loss.backward()


        ##--(2)-- REPLAYED DATA --##
        
        fixed_params = False
        mu_2, logvar_2, mu_3, mu_4 = None, None, None, None
        mu_diff, logvar_diff, x_rep, recon_batch_rep, x_atr, recon_batch_atr = None, None, None, None, None, None

        if x_ is not None:
            # In the Task-IL scenario, [y_] or [scores_] is a list and [x_] needs to be evaluated on each of them
            TaskIL = (type(y_)==list) if (y_ is not None) else (type(scores_)==list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [torch.tensor(0., device=self._device())]*n_replays
            reconL_r = [torch.tensor(0., device=self._device())]*n_replays
            variatL_r = [torch.tensor(0., device=self._device())]*n_replays
            predL_r = [torch.tensor(0., device=self._device())]*n_replays
            distilL_r = [torch.tensor(0., device=self._device())]*n_replays
            contrL_r = [torch.tensor(0., device=self._device())]*n_replays
            diffL_r = [torch.tensor(0., device=self._device())]*n_replays
            diffL_2_r = [torch.tensor(0., device=self._device())]*n_replays
            diffL_3_r = [torch.tensor(0., device=self._device())]*n_replays
            recon_repL_r = [torch.tensor(0., device=self._device())]*n_replays
            recon_atrL_r = [torch.tensor(0., device=self._device())]*n_replays
            ###lym
            ss_loss_r = [torch.tensor(0., device=self._device())] * n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)  ## Used for Class-IL...
            if (not type(x_)==list) and (self.mask_dict is None) and (not (self.dg_gates and TaskIL)):
                # -if needed in the decoder-gates, find class-tensor [y_predicted]
                y_predicted = None
                if self.dg_gates and self.dg_type=="class":
                    if y_[0] is not None:
                        y_predicted = y_[0]
                    else: ## Used for Class-IL...
                        y_predicted = F.softmax(scores_[0] / self.KD_temp, dim=1)
                        if y_predicted.size(1) < self.classes:
                            # in case of Class-IL, add zeros at the end:
                            n_batch = y_predicted.size(0)
                            zeros_to_add = torch.zeros(n_batch, self.classes - y_predicted.size(1))
                            zeros_to_add = zeros_to_add.to(self._device())
                            y_predicted = torch.cat([y_predicted, zeros_to_add], dim=1)

                # -pre-processing (if 'hidden' and [replay_not_hidden] is provided as True)
                x_temp_ = self.convE(x_) if self.hidden and replay_not_hidden else x_
                # -run full model
                gate_input = (tasks_ if self.dg_type=="task" else y_predicted) if self.dg_gates else None
                recon_batch, y_hat_all, mu, logvar, z, proj_z = self(x_temp_, gate_input=gate_input, full=True, use_views=use_views, 
                                                                     batch_size=batch_size_replay, current=False)


                #### Start of main-section modifications, all of this is new code ####

                if self.contrastive and contrast_replayed:
                    proj_z1, proj_z2 = torch.split(proj_z, [batch_size_replay, batch_size_replay], dim=0)
                    proj_z = torch.cat([proj_z1.unsqueeze(1), proj_z2.unsqueeze(1)], dim=1)
                    p1 = self.predictor(proj_z1)
                    p2 = self.predictor(proj_z2)
                    proj_z1.detach()  # stop gradient
                    proj_z2.detach()
                    x_temp_, x_ = x_temp_[:batch_size_replay], x_[:batch_size_replay]

                if top_scores_ is not None:
                    diff = True
                    rep2, averaged = True, False
                    keep_inds, similarity = None, None

                    sc_size = act_sc_size = list(top_scores_.shape)

                    specific_classes_0 = torch.reshape(top_scores_[:,0], (-1,))
                    specific_classes_1 = torch.reshape(top_scores_[:,1], (-1,))
                    specific_classes_2 = torch.reshape(top_scores_[:,2], (-1,)) if (sc_size[1] > 2) else None
                    specific_classes_3 = torch.reshape(top_scores_[:,3], (-1,)) if (sc_size[1] > 3) else None

                    if self.use_rep_factor:
                        rep_f = self.rep_factor

                        # Check probabilities...
                        y_probabilities = F.softmax(scores_[0], dim=1)
                        sc_0 = torch.reshape(specific_classes_0, (-1,1)).expand(-1, sc_size[0])
                        y_probs = torch.gather(y_probabilities, 1, sc_0)[:, 0]
                        sc_1 = torch.reshape(specific_classes_1, (-1,1)).expand(-1, sc_size[0])
                        y_probs_1 = torch.gather(y_probabilities, 1, sc_1)[:, 0]
                        sc_2 = torch.reshape(specific_classes_2, (-1,1)).expand(-1, sc_size[0]) if (specific_classes_2 is not None) else None
                        y_probs_2 = torch.gather(y_probabilities, 1, sc_2)[:, 0] if (specific_classes_2 is not None) else None
                        sc_3 = torch.reshape(specific_classes_3, (-1,1)).expand(-1, sc_size[0]) if (specific_classes_3 is not None) else None
                        y_probs_3 = torch.gather(y_probabilities, 1, sc_3)[:, 0] if (specific_classes_3 is not None) else None

                        # samples_to_use = torch.where(y_probs < (rep_f * y_probs_1))[0]
                        # samples_to_use_2 = torch.where(y_probs < (rep_f * y_probs_2))[0] if (specific_classes_2 is not None) else None
                        # samples_to_use_3 = torch.where(y_probs < (rep_f * y_probs_3))[0] if (specific_classes_3 is not None) else None
                        ###lym
                        samples_to_use = torch.where(top_threshold < y_probs_1)[0]
                        samples_to_use_2 = torch.where(top_threshold < y_probs_2)[0] if (specific_classes_2 is not None) else None
                        samples_to_use_3 = torch.where(top_threshold < y_probs_3)[0] if (specific_classes_3 is not None) else None
                    else:
                        samples_to_use = None

                    if (samples_to_use is None) or (samples_to_use.nelement() > 0):
                        if samples_to_use is not None:
                            mu_diff = mu[samples_to_use]
                            logvar_diff = logvar[samples_to_use]
                            x_comp = x_temp_[samples_to_use] if not (self.recon_repulsion or self.recon_attraction) else x_temp_
                            specific_classes_0 = specific_classes_0[samples_to_use] if not (self.recon_repulsion or self.recon_attraction) else specific_classes_0
                            specific_classes_1 = specific_classes_1[samples_to_use]
                            if not rep2:
                                if (samples_to_use_2 is not None) and (samples_to_use_2.nelement() > 0):
                                    mu_diff_3 = mu[samples_to_use_2]
                                    logvar_diff_3 = logvar[samples_to_use_2]
                                    specific_classes_2 = specific_classes_2[samples_to_use_2]
                                    mu_3, logvar_3 = self.sample(batch_size_replay, specific_classes=specific_classes_2, only_z=True)
                                    mu_3, logvar_3 = (mu_diff_3, mu_3), (logvar_diff_3, logvar_3)
                                if (samples_to_use_3 is not None) and (samples_to_use_3.nelement() > 0):
                                    mu_diff_4 = mu[samples_to_use_3]
                                    logvar_diff_4 = logvar[samples_to_use_3]
                                    specific_classes_3 = specific_classes_3[samples_to_use_3]
                                    mu_4, logvar_4 = self.sample(batch_size_replay, specific_classes=specific_classes_3, only_z=True)
                                    mu_4, logvar_4 = (mu_diff_4, mu_4), (logvar_diff_4, logvar_4)
                        elif not rep2:
                            if specific_classes_2 is not None:
                                mu_3, logvar_3 = self.sample(batch_size_replay, specific_classes=specific_classes_2, only_z=True)
                            elif specific_classes_3 is not None:
                                mu_3, logvar_3 = self.sample(batch_size_replay, specific_classes=specific_classes_2, only_z=True)
                                mu_4, logvar_4 = self.sample(batch_size_replay, specific_classes=specific_classes_3, only_z=True)
                        mu_2, logvar_2 = self.sample(batch_size_replay, specific_classes=specific_classes_1, only_z=True)
                    else:
                        diff = False

                    uniq_sc_0 = torch.unique(specific_classes_0)
                    inds_sc_0 = []

                    if rep2 and ((samples_to_use is None) or (samples_to_use.nelement() > 0)):
                        mean_mu_0 = []
                        mean_logvar_0 = []
                        mean_x = []
                        if uniq_sc_0.nelement() > 0:
                            for i, uniq_c in enumerate(uniq_sc_0):
                                inds = torch.where(specific_classes_0==uniq_sc_0[i])[0]
                                if averaged:
                                    if not self.use_rep_factor:
                                        mean_mu_0.append(torch.reshape(torch.mean(mu[inds], dim=0), (1,-1)))
                                        mean_logvar_0.append(torch.log(torch.reshape(torch.sum(torch.exp(logvar[inds]), dim=0), (1,-1)) / (inds.nelement()**2)))
                                    else:
                                        mean_mu_0.append(torch.reshape(torch.mean(mu_diff[inds], dim=0), (1,-1)))
                                        mean_logvar_0.append(torch.log(torch.reshape(torch.sum(torch.exp(logvar_diff[inds]), dim=0), (1,-1)) / (inds.nelement()**2)))

                                elif (self.recon_repulsion or self.recon_attraction) and self.recon_rep_averaged:
                                    if not self.use_rep_factor:
                                        mean_x.append(torch.mean(x_temp_[inds], dim=0))
                                    else:
                                        mean_x.append(torch.mean(x_comp[inds], dim=0))
                                else:
                                    r_ind = np.random.choice(np.arange(inds.nelement()), 1)[0]
                                    inds_sc_0.append(inds[r_ind])

                            mean_mu_0 = torch.cat(mean_mu_0, dim=0) if averaged else None
                            mean_logvar_0 = torch.cat(mean_logvar_0, dim=0) if averaged else None
                            inds_sc_0 = torch.tensor(inds_sc_0, device=self._device()) if not averaged else None

                            keep_inds = []
                            def map_inds(a, uniq):
                                uniq_ind = torch.where(uniq==a)[0]
                                if uniq_ind.nelement() < 1:
                                    keep_inds.append(0)
                                    uniq_ind = torch.tensor(np.random.choice(np.arange(uniq.nelement()), 1)[0], device=self._device())
                                else:
                                    keep_inds.append(1)
                                return inds_sc_0[uniq_ind]

                            def map_x(a, uniq, rep=True):
                                uniq_ind = torch.where(uniq==a)[0]
                                if uniq_ind.nelement() < 1:
                                    if rep==True:
                                        keep_inds.append(0)
                                    uniq_ind = torch.tensor(np.random.choice(np.arange(uniq.nelement()), 1)[0], device=self._device())
                                elif rep==True:
                                    keep_inds.append(1)
                                return mean_x[uniq_ind]

                            def map_mus(a, uniq):
                                uniq_ind = torch.where(uniq==a)[0]
                                if uniq_ind.nelement() < 1:
                                    keep_inds.append(0)
                                    uniq_ind = torch.tensor(np.random.choice(np.arange(uniq.nelement()), 1)[0], device=self._device())
                                    return torch.reshape(mean_mu_0[uniq_ind], (1,-1))
                                else:
                                    keep_inds.append(1)
                                    return mean_mu_0[uniq_ind]

                            if averaged:
                                mu_b = torch.cat(list(map(functools.partial(map_mus, uniq=uniq_sc_0), specific_classes_1)), dim=0)
                                keep_inds = [i for i, x in enumerate(keep_inds) if x == 1]
                            else:
                                if self.recon_repulsion:
                                    # Recon batch is the batch of reconstructed samples...
                                    recon_batch_rep = recon_batch if not self.use_rep_factor else recon_batch[samples_to_use]
                                    # Find indices of competing classes & create batch of competing samples, x_rep...
                                    if self.recon_rep_averaged:
                                        # Apply reconstruction repulsion loss to the average feature vector across all samples from the competing class...
                                        x_rep = torch.cat(list(map(functools.partial(map_x, uniq=uniq_sc_0), specific_classes_1)), dim=0)
                                        keep_inds = [i for i, x in enumerate(keep_inds) if x == 1]
                                    else:
                                        # Apply reconstruction repulsion loss to random samples from the competing class...
                                        inds_1 = torch.tensor(list(map(functools.partial(map_inds, uniq=uniq_sc_0), specific_classes_1)), device=self._device())
                                        x_rep = x_temp_[inds_1] if not self.use_rep_factor else x_comp[inds_1]
                                else:
                                    inds_1 = torch.tensor(list(map(functools.partial(map_inds, uniq=uniq_sc_0), specific_classes_1)), device=self._device())
                                    mu_b, logvar_b = mu[inds_1], logvar[inds_1]

                                if self.recon_attraction:
                                    # Recon batch is the batch of reconstructed samples...
                                    recon_batch_atr = recon_batch
                                    # Find indices of competing classes & create batch of competing samples, x_atr...
                                    if self.recon_rep_averaged:
                                        # Apply reconstruction repulsion loss to the average feature vector across all samples from the competing class...
                                        x_atr = torch.cat(list(map(functools.partial(map_x, uniq=uniq_sc_0, rep=False), specific_classes_0)), dim=0)
                                    else:
                                        # Apply reconstruction repulsion loss to random samples from the competing class...
                                        inds_0 = torch.tensor(list(map(functools.partial(map_inds, uniq=uniq_sc_0), specific_classes_0)), device=self._device())
                                        x_atr = x_temp_[inds_0]

                            if len(keep_inds)==0:
                                diff = False

            #### End of main-section modifications ####


            # Loop to perform each replay
            for replay_id in range(n_replays):
                #---> NOTE: pre-processing is sometimes needed for 'hidden' (as only generated replay comes as features)

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay ## Not used for Class-IL...
                if (type(x_)==list) or (self.mask_dict is not None) or (TaskIL and self.dg_gates):
                    # -if needed in the decoder-gates, find class-tensor [y_predicted]
                    y_predicted = None
                    if self.dg_gates and self.dg_type == "class":
                        if y_ is not None and y_[replay_id] is not None:
                            y_predicted = y_[replay_id]
                            # because of Task-IL, increase class-ID with number of classes before task being replayed
                            y_predicted = y_predicted + replay_id*len(active_classes[0])
                        else:
                            y_predicted = F.softmax(scores_[replay_id] / self.KD_temp, dim=1)
                            if y_predicted.size(1) < self.classes:
                                # in case of Task-IL, add zeros before and after:
                                n_batch = y_predicted.size(0)
                                zeros_to_add_before = torch.zeros(n_batch, replay_id*y_predicted.size(1))
                                zeros_to_add_before = zeros_to_add_before.to(self._device())
                                zeros_to_add_after = torch.zeros(n_batch,self.classes-(replay_id+1)*y_predicted.size(1))
                                zeros_to_add_after = zeros_to_add_after.to(self._device())
                                y_predicted = torch.cat([zeros_to_add_before, y_predicted, zeros_to_add_after], dim=1)
                    # -need to pre-process?
                    x_temp_ = x_[replay_id] if type(x_)==list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id+1)
                    x_temp_ = self.convE(x_temp_) if self.hidden and replay_not_hidden else x_temp_
                    # -run full model
                    gate_input = (tasks_[replay_id] if self.dg_type=="task" else y_predicted) if self.dg_gates else None
                    recon_batch, y_hat_all, mu, logvar, z, proj_z = self(x_temp_, full=True, gate_input=gate_input)


                # -if needed (e.g., "class" or "task" scenario), remove predictions for classes not in replayed task
                y_hat = y_hat_all if (
                        active_classes is None or y_hat_all is None
                ) else y_hat_all[:, active_classes[replay_id]]

                #### Output of loss function & combination of losses, followed by back-propagation have been heavily modified from here down to end of file ####

                # Calculate all losses
                if self.contrastive and contrast_replayed:
                    ###lym
                    ss_loss_r[replay_id] = -(criterion(p1, proj_z2).mean() + criterion(p2, proj_z1).mean()) * 0.5

                ###0822
                if (not self.repulsion) or (not diff):
                    if self.contrastive and contrast_replayed:
                        reconL_r[replay_id], variatL_r[replay_id], predL_r[replay_id], distilL_r[replay_id], contrL_r[
                            replay_id], recon_repL_r[replay_id], recon_atrL_r[replay_id] = self.loss_function(
                            x=x_temp_, y=y_[replay_id] if (y_ is not None) else None, x_recon=recon_batch, y_hat=y_hat,
                            scores=scores_[replay_id] if (scores_ is not None) else None, mu=mu, z=z, logvar=logvar,
                            allowed_classes=active_classes[replay_id] if active_classes is not None else None,
                            proj_z=proj_z, use_views=use_views, x_rep=x_rep,
                            x_recon_rep=recon_batch_rep, x_atr=x_atr, x_recon_atr=recon_batch_atr,
                            keep_inds=keep_inds if top_scores_ is not None else None
                        )
                    else:
                        reconL_r[replay_id], variatL_r[replay_id], predL_r[replay_id], distilL_r[replay_id], contrL_r[
                            replay_id], recon_repL_r[replay_id], recon_atrL_r[replay_id] = self.loss_function(
                            x=x_temp_, y=y_[replay_id] if (y_ is not None) else None, x_recon=recon_batch, y_hat=y_hat,
                            scores=scores_[replay_id] if (scores_ is not None) else None, mu=mu, z=z, logvar=logvar,
                            allowed_classes=active_classes[replay_id] if active_classes is not None else None,
                            proj_z=proj_z, use_views=use_views, x_rep=x_rep,
                            x_recon_rep=recon_batch_rep, x_atr=x_atr, x_recon_atr=recon_batch_atr,
                            keep_inds=keep_inds if top_scores_ is not None else None
                        )
                elif mu_3 is None:
                    reconL_r[replay_id], variatL_r[replay_id], diffL_r[replay_id], predL_r[replay_id], distilL_r[
                        replay_id], contrL_r[replay_id], recon_repL_r[replay_id], recon_atrL_r[
                        replay_id] = self.loss_function(
                        x=x_temp_, y=y_[replay_id] if (y_ is not None) else None, x_recon=recon_batch, y_hat=y_hat,
                        scores=scores_[replay_id] if (scores_ is not None) else None, mu=mu, z=z, logvar=logvar,
                        allowed_classes=active_classes[replay_id] if active_classes is not None else None,
                        diff=diff, mu_diff=mu_diff, logvar_diff=logvar_diff, mu_2=mu_2, logvar_2=logvar_2,
                        kl_js=self.kl_js,
                        mu_b=mu_b, logvar_b=logvar_b, keep_inds=keep_inds if top_scores_ is not None else None,
                        similarity=similarity,
                        proj_z=proj_z, use_views=use_views, x_rep=x_rep, x_recon_rep=recon_batch_rep, x_atr=x_atr,
                        x_recon_atr=recon_batch_atr)

                elif mu_4 is None:
                    reconL_r[replay_id], variatL_r[replay_id], diffL_r[replay_id], diffL_2_r[replay_id], predL_r[
                        replay_id], distilL_r[replay_id], contrL_r[replay_id], recon_repL_r[replay_id], recon_atrL_r[
                        replay_id] = self.loss_function(
                        x=x_temp_, y=y_[replay_id] if (y_ is not None) else None, x_recon=recon_batch, y_hat=y_hat,
                        scores=scores_[replay_id] if (scores_ is not None) else None, mu=mu, z=z, logvar=logvar,
                        allowed_classes=active_classes[replay_id] if active_classes is not None else None,
                        diff=diff, mu_diff=mu_diff, logvar_diff=logvar_diff, mu_2=mu_2, logvar_2=logvar_2,
                        mu_3=mu_3, logvar_3=logvar_3, kl_js=self.kl_js, use_rep_factor=self.use_rep_factor,
                        proj_z=proj_z, use_views=use_views, x_rep=x_rep,
                        x_recon_rep=recon_batch_rep, x_atr=x_atr, x_recon_atr=recon_batch_atr,
                        keep_inds=keep_inds if top_scores_ is not None else None
                    )
                else:
                    reconL_r[replay_id], variatL_r[replay_id], diffL_r[replay_id], diffL_2_r[replay_id], diffL_3_r[
                        replay_id], predL_r[replay_id], distilL_r[replay_id], contrL_r[replay_id], recon_repL_r[
                        replay_id], recon_atrL_r[replay_id] = self.loss_function(
                        x=x_temp_, y=y_[replay_id] if (y_ is not None) else None, x_recon=recon_batch, y_hat=y_hat,
                        scores=scores_[replay_id] if (scores_ is not None) else None, mu=mu, z=z, logvar=logvar,
                        allowed_classes=active_classes[replay_id] if active_classes is not None else None,
                        diff=diff, mu_diff=mu_diff, logvar_diff=logvar_diff, mu_2=mu_2, logvar_2=logvar_2,
                        mu_3=mu_3, logvar_3=logvar_3, mu_4=mu_4, logvar_4=logvar_4, kl_js=self.kl_js,
                        use_rep_factor=self.use_rep_factor,
                        proj_z=proj_z, use_views=use_views, x_rep=x_rep, x_recon_rep=recon_batch_rep,
                        x_recon_atr=recon_batch_atr, keep_inds=keep_inds
                    )

                # Weigh losses as requested ###
                loss_replay[replay_id] = self.lamda_rcl*reconL_r[replay_id] + self.lamda_vl*variatL_r[replay_id]
                if self.replay_targets=="hard":
                    loss_replay[replay_id] += self.lamda_pl*predL_r[replay_id]
                elif self.replay_targets=="soft":
                    loss_replay[replay_id] += self.lamda_pl*distilL_r[replay_id]
                
                #### Weighting the difference loss...
                if self.repulsion and diff:
                    if mu_3 is None:
                        loss_replay[replay_id] += self.lamda_rep * diffL_r[replay_id]
                    elif mu_4 is None:
                        loss_replay[replay_id] += self.lamda_rep * diffL_r[replay_id]
                        loss_replay[replay_id] += self.lamda_rep * diffL_2_r[replay_id]
                    else:
                        loss_replay[replay_id] += self.lamda_rep * diffL_r[replay_id]
                        loss_replay[replay_id] += self.lamda_rep * diffL_2_r[replay_id]
                        loss_replay[replay_id] += self.lamda_rep * diffL_3_r[replay_id]

                if self.recon_repulsion and (x_rep is not None) and (recon_repL_r[replay_id] is not None):
                    loss_replay[replay_id] += self.lamda_recon_rep * recon_repL_r[replay_id]

                if self.recon_attraction and (x_atr is not None) and (recon_atrL_r[replay_id] is not None):
                    loss_replay[replay_id] += self.lamda_recon_atr * recon_atrL_r[replay_id]

                ####

                # If task-specific mask, backward pass needs to be performed before next task-mask is applied
                if self.mask_dict is not None:
                    weighted_replay_loss_this_task = (1-rnt) * loss_replay[replay_id] / n_replays
                    #### Before optimisation step, set requires_grad = False for encoder &
                    #### projection head...
                    if self.contrastive and fixed_params is False:
                        fixed_params = True
                        for param in self.parameters():
                            param.requires_grad = True
                        if not self.use_attention:
                            for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters()):
                                param.requires_grad = False
                        else:
                            for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters(),
                                           self.multihead_attn.parameters(), self.E_attn.parameters()):
                                param.requires_grad = False

                    # Update gradients...
                    weighted_replay_loss_this_task.backward()
        
        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay)/n_replays
        loss_total = loss_replay if (x is None) else (loss_cur if x_ is None else rnt*loss_cur+(1-rnt)*loss_replay)

        #### Calculate total contrastive loss...
        if self.contrastive:
            loss_replay_contr = None if (x_ is None) else sum(contrL_r) / n_replays
            ss_loss_r = None if (x_ is None) else sum(ss_loss_r) / n_replays
            if contrast_current:
                loss_total_contr = loss_replay_contr if (x is None) else (
                    contrL if x_ is None else rnt * contrL + (1 - rnt) * loss_replay_contr)

                loss_total_ssl = ss_loss_r if (x is None) else (
                    ss_loss if x_ is None else 0.2 * ss_loss + (1 - 0.2) * ss_loss_r)
            else:
                loss_total_contr = loss_replay_contr
                loss_total_ssl = ss_loss_r

        ##--(3)-- ALLOCATION LOSSES --##
        
        if self.contrastive and fixed_params is False:
            fixed_params = True
            for param in self.parameters():
                param.requires_grad = True
            if not self.use_attention:
                for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters(), ):
                    param.requires_grad = False
            else:
                for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters(),
                                   self.multihead_attn.parameters(), self.E_attn.parameters()):
                    param.requires_grad = False

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c>0:
            loss_total += self.si_c * surrogate_loss

        # Add EWC-loss
        ewc_loss = self.ewc_loss()
        if self.ewc_lambda>0:
            loss_total += self.ewc_lambda * ewc_loss

        # Backpropagate errors (if not yet done)
        if (self.mask_dict is None) or (x_ is None):
            #### Before optimisation step, set requires_grad = False for encoder &
            #### projection head...
            if self.contrastive and fixed_params is False:
                fixed_params = True
                for param in self.parameters():
                    param.requires_grad = True
                if not self.use_attention:
                    for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters()):
                        param.requires_grad = False
                else:
                    for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters(),
                                       self.multihead_attn.parameters(), self.E_attn.parameters()):
                        param.requires_grad = False

            # Update gradients...
            loss_total.backward(retain_graph=True)

        #### Before encoder optimisation step, set requires_grad = True for encoder &
        #### encoder & projection head, and requires_grad = False for
        #### everything else...
        if self.contrastive and (loss_total_contr is not None):
            for param in self.parameters():
                param.requires_grad = False
            #for param in chain(self.convE.parameters(), self.fcE.parameters(), self.fcProj.parameters()):
            if not self.use_attention:
                for param in chain(self.fcE.parameters(), self.fcProj.parameters(), self.predictor.parameters()):
                    param.requires_grad = True
            else:
                for param in chain(self.fcE.parameters(), self.fcProj.parameters(), self.predictor.parameters(),
                                   self.multihead_attn.parameters(), self.E_attn.parameters()):
                    param.requires_grad = True
    
            #### Update encoder gradients...
            if self.simsiam:
                loss_total_ssl.backward()
            else:
                loss_total_contr.backward()

            
            #### Take encoder optimization-step...
            self.E_optimizer.step()

        if self.contrastive:
            for param in self.parameters():
                param.requires_grad = True
            if not self.use_attention:
                for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters()):
                    param.requires_grad = False
            else:
                for param in chain(self.convE.parameters(), self.fcProj.parameters(), self.predictor.parameters(),
                                   self.multihead_attn.parameters(), self.E_attn.parameters()):
                    param.requires_grad = False
            
            # Take optimization-step
            self.optimizer.step()

        else:
            # Take optimization-step
            self.optimizer.step()

        # Return the dictionary with different training-loss split in categories ###
        return {
            'loss_total': loss_total.item(), 'precision': precision,
            'recon': reconL.item() if x is not None else 0,
            'variat': variatL.item() if x is not None and variatL!=0 else 0,
            'pred': predL.item() if x is not None else 0,
            'contr': contrL.item() if (x is not None) and (self.contrastive) and (contrast_current) else 0,
            'recon_r': sum(reconL_r).item()/n_replays if x_ is not None else 0,
            'variat_r': sum(variatL_r).item()/n_replays if x_ is not None else 0,
            'diff_r': sum(diffL_r).item()/n_replays if (x_ is not None) and (self.repulsion) and diff else 0,
            'diff_2_r': sum(diffL_2_r).item()/n_replays if (x_ is not None) and (self.repulsion) and diff and (mu_3 is not None) else 0,
            'diff_3_r': sum(diffL_3_r).item()/n_replays if (x_ is not None) and (self.repulsion) and diff  and (mu_4 is not None) else 0,
            'recon_repL_r': sum(recon_repL_r).item()/n_replays if (x_ is not None) and (self.recon_repulsion) and (x_rep is not None) and (recon_repL_r[0] is not None) else 0,
            'recon_atrL_r': sum(recon_atrL_r).item()/n_replays if (x_ is not None) and (self.recon_attraction) and (x_atr is not None) and (recon_atrL_r[0] is not None)  else 0,
            'pred_r': sum(predL_r).item()/n_replays if x_ is not None else 0,
            'distil_r': sum(distilL_r).item()/n_replays if x_ is not None else 0,
            'contr_r': sum(contrL_r).item()/n_replays if (x_ is not None) and (self.contrastive) and (contrast_replayed) else 0,
            'ewc': ewc_loss.item(), 'si_loss': surrogate_loss.item(),
            'loss_total_ssl': loss_total_ssl if self.simsiam else 0,
        }
