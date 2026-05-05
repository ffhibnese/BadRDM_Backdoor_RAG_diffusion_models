import torch
from torchvision.models import densenet121, resnet50, vgg16
from diffusers.optimization import get_cosine_schedule_with_warmup

try:
    from backdoor_attack.dataset import CC3M, LaionAesthetics65
    from backdoor_attack.losses import losses
except ImportError:
    from dataset import CC3M, LaionAesthetics65
    from losses import losses

def get_optimizer(config, model):
    weight_decay_parameters = []
    no_weight_decay_parameters = []
    for name, parameter in model.named_parameters():
        if(all(key not in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
            weight_decay_parameters.append(parameter)
            
        if(any(key in name for key in ["bn", "ln", "bias", "logit_scale"]) and parameter.requires_grad):
            no_weight_decay_parameters.append(parameter)
    adam_config = config.optimizer.AdamW
    optimizer = torch.optim.AdamW(
        [{"params": no_weight_decay_parameters, "weight_decay": 0}, {"params": weight_decay_parameters, "weight_decay": adam_config.weight_decay}],
        lr=adam_config.lr,
        betas=(adam_config.betas[0], adam_config.betas[1]),
        eps=adam_config.eps
    )
    
    return optimizer

def get_scheduler(config, optimizer, num_training_steps):
    # sch_config = config.lr_scheduler.MultiStepLR
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  milestones=sch_config.milestones,
    #                                                  gamma=sch_config.gamma)
    warmup_steps = int(config.train.get("warmup_steps", 500)) if "train" in config else 500
    warmup_steps = min(warmup_steps, max(0, num_training_steps - 1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    
    return scheduler

def get_loss_func(loss_type, flatten=True, t=0.1):
    loss_func = None
    if loss_type == 'SimilarityLoss':
        loss_func = losses.SimilarityLoss(flatten=flatten)
    elif loss_type == 'ContrastLoss':
        loss_func = losses.ContrastLoss(t=t)
    elif loss_type == 'ClipLoss':
        loss_func = losses.ClipLoss()
    
    return loss_func

def get_dataset(config, transform):
    if config.dataset == 'laion':
        dataset = LaionAesthetics65(transform, data_root=config.image_root)
    elif config.dataset == 'cc3m':
        dataset = CC3M(transform, config.image_root, anno_file=config.anno_file, max_words=config.define.max_words)
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    return dataset

def get_classifier(clf, device):
    if clf == 'vgg':
        classifier = vgg16(pretrained=True).to(device).eval()
    elif clf == 'densenet':
        classifier = densenet121(pretrained=True).to(device).eval()
    
    return classifier
