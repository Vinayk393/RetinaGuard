"""
RetinaGuard — src package
=========================
Modular source code for the RetinaGuard diabetic retinopathy
classification system.

Modules
-------
utils       : set_seed, make_output_dirs, save_json, setup_logger, get_device
models      : BaselineCNN, build_efficientnet_b0, build_resnet50, build_model
preprocess  : apply_clahe, get_transforms, RetinopathyDataset,
              compute_class_weights, make_dataloaders
train       : FocalLoss, get_ce_loss, get_weighted_ce_loss, get_focal_loss,
              train_one_epoch, evaluate, train_model
evaluate    : compute_metrics, predict_with_tta, optimise_threshold,
              print_metrics_table
interpret   : get_cam_target_layer, run_gradcam, run_ieee_gradcam,
              analyse_per_class_recall, print_interpretation

Entry point
-----------
run_experiment.py
    CLI runner: python src/run_experiment.py --config configs/ce.yaml
"""

from .utils      import set_seed, make_output_dirs, save_json, load_json, setup_logger, get_device
from .models     import (BaselineCNN, build_efficientnet_b0, build_resnet50,
                         build_model, count_params, print_model_summary)
from .preprocess import (apply_clahe, get_transforms, RetinopathyDataset,
                         compute_class_weights, make_dataloaders)
from .train      import (get_ce_loss, get_weighted_ce_loss, FocalLoss, get_focal_loss,
                         train_one_epoch, evaluate, train_model)
from .evaluate   import (compute_metrics, predict_with_tta, optimise_threshold,
                         print_metrics_table)
from .interpret  import (get_cam_target_layer, run_gradcam, run_ieee_gradcam,
                         analyse_per_class_recall, print_interpretation)

__version__ = '1.0.0'
