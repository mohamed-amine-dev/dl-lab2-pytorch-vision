"""Rerun the two fine-tuned models with a larger LR for the freshly initialised head."""
from src import train_finetune

if __name__ == "__main__":
    train_finetune.main(model_name="vgg16", epochs=2, batch_size=32, lr=3e-3,
                        image_size=64, train_subset=10000)
    train_finetune.main(model_name="alexnet", epochs=3, batch_size=64, lr=3e-3,
                        image_size=64, train_subset=20000)
