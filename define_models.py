import utils
from utils import checkattr

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining auto-encoder model
def define_autoencoder(args, config, device, generator=False, convE=None):
    # -import required model
    from models.vae import AutoEncoder
    # -create model
    if (hasattr(args, "depth") and args.depth > 0):
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -conv-layers
            conv_type=args.conv_type, depth=args.g_depth if generator  and hasattr(args, 'g_depth') else args.depth,
            start_channels=args.channels, reducing_layers=args.rl, conv_bn=(args.conv_bn=="yes"), conv_nl=args.conv_nl,
            num_blocks=args.n_blocks, convE=convE, global_pooling=False if generator else checkattr(args, 'gp'),
            # -fc-layers
            fc_layers=args.g_fc_lay if generator and hasattr(args, 'g_fc_lay') else args.fc_lay,
            fc_units=args.g_fc_uni if generator and hasattr(args, 'g_fc_uni') else args.fc_units,
            h_dim=args.g_h_dim if generator and hasattr(args, 'g_h_dim') else args.h_dim,
            fc_drop=0 if generator else args.fc_drop, fc_bn=(args.fc_bn=="yes"), fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior if hasattr(args, "prior") else "standard",
            n_modes=args.n_modes if hasattr(args, "prior") else 1,
            per_class=args.per_class if hasattr(args, "prior") else False,
            z_dim=args.g_z_dim if generator  and hasattr(args, 'g_z_dim') else args.z_dim,
            # -decoder
            hidden=checkattr(args, 'hidden'),
            recon_loss=args.recon_loss, network_output="none" if checkattr(args, "normalize") else "sigmoid",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
            dg_gates=utils.checkattr(args, 'dg_gates'), dg_type=args.dg_type if hasattr(args, 'dg_type') else "task",
            dg_prop=args.dg_prop if hasattr(args, 'dg_prop') else 0.,
            tasks=args.tasks if hasattr(args, 'tasks') else None,
            scenario=args.scenario if hasattr(args, 'scenario') else None, device=device,
            # -classifier
            classifier=False if generator else True,
            classify_opt=args.classify if hasattr(args, "classify") else "beforeZ",
            # -training-specific components
            lamda_rcl=1. if not hasattr(args, 'rcl') else args.rcl,
            lamda_vl=1. if not hasattr(args, 'vl') else args.vl,
            lamda_pl=(0. if generator else 1.) if not hasattr(args, 'pl') else args.pl,
            ####
            lamda_rep=1e-6 if not hasattr(args, 'repl') else args.repl,
            repulsion=False if not hasattr(args, 'repulsion') else args.repulsion,
            kl_js='js' if not hasattr(args, 'kl_js') else args.kl_js,
            use_rep_factor=False if not hasattr(args, 'use_rep_factor') else args.use_rep_factor,
            rep_factor=20 if not hasattr(args, 'rep_factor') else args.rep_factor,
            n_rep=1 if not hasattr(args, 'n_rep') else args.n_rep,
            apply_mask=False if not hasattr(args, 'apply_mask') else args.apply_mask,
            param_tuning=False if not hasattr(args, 'tuning') else args.tuning,
            contrastive=False if not hasattr(args, 'contrastive') else args.contrastive,
            c_temp=1.0 if not hasattr(args, 'c_temp') else args.c_temp,
            contr_lr=1e-8 if not hasattr(args, 'contr_lr') else args.contr_lr,
            c_drop=0.5 if not hasattr(args, 'c_drop') else args.c_drop,
            contr_not_hidden=True if hasattr(args, 'contrastive') else checkattr(args, 'contr_not_hidden'),
            recon_repulsion=False if not hasattr(args, 'recon_repulsion') else args.recon_repulsion,
            recon_rep_averaged=False if not hasattr(args, 'recon_rep_averaged') else args.recon_rep_averaged,
            lamda_recon_rep=1e-6 if not hasattr(args, 'recon_repl') else args.recon_repl,
            recon_attraction=False if not hasattr(args, 'recon_attraction') else args.recon_attraction,
            lamda_recon_atr=1e-6 if not hasattr(args, 'recon_atrl') else args.recon_atrl,
            contr_scores=False if not hasattr(args, 'contr_scores') else args.contr_scores,
            contr_hard=False if not hasattr(args, 'contr_hard') else args.contr_hard,
            simsiam=False if not hasattr(args, 'simsiam') else args.simsiam,
            lamda_ssl=1e-6 if not hasattr(args, 'lamda_ssl') else args.lamda_ssl,
            attention=False if not hasattr(args, 'attention') else args.attention,
            ma=False if not hasattr(args, 'ma') else args.ma,
            ma_drop=0.1 if not hasattr(args, 'ma_drop') else args.ma_drop,
            ###lym
            ####
        ).to(device)
    else:
        model = AutoEncoder(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -fc-layers
            fc_layers=args.g_fc_lay if generator and hasattr(args, 'g_fc_lay') else args.fc_lay,
            fc_units=args.g_fc_uni if generator and hasattr(args, 'g_fc_uni') else args.fc_units,
            h_dim=args.g_h_dim if generator and hasattr(args, 'g_h_dim') else args.h_dim,
            fc_drop=0 if generator else args.fc_drop, fc_bn=(args.fc_bn=="yes"), fc_nl=args.fc_nl, excit_buffer=True,
            # -prior
            prior=args.prior if hasattr(args, "prior") else "standard",
            n_modes=args.n_modes if hasattr(args, "prior") else 1,
            per_class=args.per_class if hasattr(args, "prior") else False,
            z_dim=args.g_z_dim if generator and hasattr(args, 'g_z_dim') else args.z_dim,
            # -decoder
            recon_loss=args.recon_loss, network_output="none" if checkattr(args, "normalize") else "sigmoid",
            deconv_type=args.deconv_type if hasattr(args, "deconv_type") else "standard",
            dg_gates=utils.checkattr(args, 'dg_gates'), dg_type=args.dg_type if hasattr(args, 'dg_type') else "task",
            dg_prop=args.dg_prop if hasattr(args, 'dg_prop') else 0.,
            tasks=args.tasks if hasattr(args, 'tasks') else None,
            scenario=args.scenario if hasattr(args, 'scenario') else None, device=device,
            # -classifier
            classifier=False if generator else True,
            classify_opt=args.classify if hasattr(args, "classify") else "beforeZ",
            # -training-specific components
            lamda_rcl=1. if not hasattr(args, 'rcl') else args.rcl,
            lamda_vl=1. if not hasattr(args, 'vl') else args.vl,
            lamda_pl=(0. if generator else 1.) if not hasattr(args, 'pl') else args.pl,
            ####
            lamda_rep=1e-6 if not hasattr(args, 'repl') else args.repl,
            repulsion=False if not hasattr(args, 'repulsion') else args.repulsion,
            kl_js='js' if not hasattr(args, 'kl_js') else args.kl_js,
            use_rep_factor=False if not hasattr(args, 'use_rep_factor') else args.use_rep_factor,
            rep_factor=20 if not hasattr(args, 'rep_factor') else args.rep_factor,
            n_rep=1 if not hasattr(args, 'n_rep') else args.n_rep,
            param_tuning=False if not hasattr(args, 'tuning') else args.tuning,
            contrastive=False if not hasattr(args, 'contrastive') else args.contrastive,
            c_temp=1.0 if not hasattr(args, 'c_temp') else args.c_temp,
            contr_lr=1e-8 if not hasattr(args, 'contr_lr') else args.contr_lr,
            c_drop=0.5 if not hasattr(args, 'c_drop') else args.c_drop,
            recon_repulsion=False if not hasattr(args, 'recon_repulsion') else args.recon_repulsion,
            recon_rep_averaged=False if not hasattr(args, 'recon_rep_averaged') else args.recon_rep_averaged,
            lamda_recon_rep=1e-6 if not hasattr(args, 'recon_repl') else args.recon_repl,
            recon_attraction=False if not hasattr(args, 'recon_attraction') else args.recon_attraction,
            lamda_recon_atr=1e-6 if not hasattr(args, 'recon_atrl') else args.recon_atrl,
            contr_scores=False if not hasattr(args, 'contr_scores') else args.contr_scores,
            contr_hard=False if not hasattr(args, 'contr_hard') else args.contr_hard,
            simsiam=False if not hasattr(args, 'simsiam') else args.simsiam,
            lamda_ssl=1e-6 if not hasattr(args, 'lamda_ssl') else args.lamda_ssl,
            attention=False if not hasattr(args, 'attention') else args.attention,
            ma=False if not hasattr(args, 'ma') else args.ma,
            ma_drop=0.1 if not hasattr(args, 'ma_drop') else args.ma_drop,
            ###lym
            ####
        ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for defining classifier model
def define_classifier(args, config, device):
    # -import required model
    from models.classifier import Classifier
    # -create model
    if (hasattr(args, "depth") and args.depth>0):
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -conv-layers
            conv_type=args.conv_type, depth=args.depth, start_channels=args.channels, reducing_layers=args.rl,
            num_blocks=args.n_blocks, conv_bn=True if args.conv_bn=="yes" else False, conv_nl=args.conv_nl,
            global_pooling=checkattr(args, 'gp'),
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
            # -training-specific components
            hidden=checkattr(args, 'hidden'),
        ).to(device)
    else:
        model = Classifier(
            image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
            # -fc-layers
            fc_layers=args.fc_lay, fc_units=args.fc_units, h_dim=args.h_dim,
            fc_drop=args.fc_drop, fc_bn=True if args.fc_bn=="yes" else False, fc_nl=args.fc_nl, excit_buffer=True,
        ).to(device)
    # -return model
    return model

##-------------------------------------------------------------------------------------------------------------------##

## Function for (re-)initializing the parameters of [model]
def init_params(model, args):
    # - reinitialize all parameters according to default initialization
    model.apply(utils.weight_reset)
    # - initialize parameters according to chosen custom initialization (if requested)
    if hasattr(args, 'init_weight') and not args.init_weight=="standard":
        utils.weight_init(model, strategy="xavier_normal")
    if hasattr(args, 'init_bias') and not args.init_bias=="standard":
        utils.bias_init(model, strategy="constant", value=0.01)
    # - use pre-trained weights (either for full model or just in conv-layers)?
    if utils.checkattr(args, "pre_convE") and hasattr(model, 'depth') and model.depth>0:
        load_name = model.convE.name if (
            not hasattr(args, 'convE_ltag') or args.convE_ltag=="none"
        ) else "{}-{}".format(model.convE.name, args.convE_ltag)
        print('load_checkpoint',1)
        utils.load_checkpoint(model.convE, model_dir=args.m_dir, name=load_name)
    if utils.checkattr(args, "pre_convD") and hasattr(model, 'convD') and model.depth>0:
        utils.load_checkpoint(model.convD, model_dir=args.m_dir)
    return model

##-------------------------------------------------------------------------------------------------------------------##
