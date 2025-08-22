import argparse
import torch
import os 
import sys
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=60)

    args = parser.parse_args()

    lr = 0.0004 # learning rate
    # l1_coeff = 0.00014 # l1 sparsity regularization coefficient
    l1_coeff = 0.000014 # l1 sparsity regularization coefficient
    hook_point_layer = args.layer

    # torch.set_default_device(f"cuda:{args.device}")

    cfg = LanguageModelSAERunnerConfig(
        hook_point_layer = hook_point_layer,
        hook_point = f"mlp_input.{hook_point_layer}",
        d_in = 2880,
        # dataset_path = "HuggingFaceFW/fineweb",
        dataset_path = "HuggingFaceTB/cosmopedia",
        is_dataset_tokenized=False,
        model_name='openai/gpt-oss-20b',
        is_transcoder = True,
        out_hook_point = f"mlp_output.{hook_point_layer}",
        out_hook_point_layer = hook_point_layer,
        d_out = 2880,
        
        # SAE Parameters
        expansion_factor = args.expansion_factor,
        b_dec_init_method = "mean",
        
        # Training Parameters
        lr = lr,
        l1_coefficient = l1_coeff,
        lr_scheduler_name="constantwithwarmup",
        train_batch_size = 4096,
        # train_batch_size = 2048,
        context_size = 512,
        # lr_warm_up_steps=5000,
        lr_warm_up_steps=500,
        
        # Activation Store Parameters
        n_batches_in_buffer = 32,
        # n_batches_in_buffer = 128,
        total_training_tokens = 1_000_000 * args.tokens,
        store_batch_size = 32,
        
        # Dead Neurons and Sparsity
        use_ghost_grads=True,
        feature_sampling_method = 'anthropic',
        feature_sampling_window = 1000,
        resample_batches=128,
        dead_feature_window=1000,
        dead_feature_threshold = 1e-8,

        # WANDB
        log_to_wandb = False,

        # Comet
        log_to_comet = True,
        
        # Misc
        use_tqdm = True,
        device = f"cuda:{args.device}",
        seed = 42,
        n_checkpoints = 5,
        checkpoint_path = "gpt-oss-20b-transcoders", # change as you please
        dtype = torch.float32,
    )

    print(f"About to start training with lr {lr} and l1 {l1_coeff}")
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(cfg)

    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

    # train SAE
    sparse_autoencoder = train_sae_on_language_model(
        model, sparse_autoencoder, activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size = cfg.train_batch_size,
        feature_sampling_method = cfg.feature_sampling_method,
        feature_sampling_window = cfg.feature_sampling_window,
        feature_reinit_scale = cfg.feature_reinit_scale,
        dead_feature_threshold = cfg.dead_feature_threshold,
        dead_feature_window=cfg.dead_feature_window,
        use_wandb = cfg.log_to_wandb,
        wandb_log_frequency = cfg.wandb_log_frequency,
        use_comet = cfg.log_to_comet
    )

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/final_{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.save_model(path)

