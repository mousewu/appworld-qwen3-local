"""Full-parameter Qwen3 SFT using the shared completion-only training pipeline."""
from train_lora import main

if __name__ == '__main__':
    main(default_mode='full')
