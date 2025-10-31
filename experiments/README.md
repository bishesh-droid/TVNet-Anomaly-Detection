# experiments/

This folder stores runtime outputs for experiments.

Sub-folders:
- logs/         : textual run logs
- checkpoints/  : saved model checkpoints (.pth)
- results/      : tabular metrics, JSONs, plots

Usage:
- Set save_dir and log_dir in the config files.
- Trainer scripts call utils.logger.save_checkpoint to write checkpoints here.
