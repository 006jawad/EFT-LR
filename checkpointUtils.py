import os
import json
import csv
import torch
import transformers
from safetensors.torch import load_file
from peft import set_peft_model_state_dict

class CSVLoggerCallback(transformers.TrainerCallback):
    """
    A callback for logging training metrics to a CSV file during training with Hugging Face's Trainer.

    This callback logs key metrics such as epoch, step, loss, evaluation loss, gradient norm, 
    and learning rate at each logging step. It ensures logging happens only in the main process 
    in distributed training setups.

    Attributes:
        csv_file_path (str): Path to the CSV file where logs are stored.
    """

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            with open(self.csv_file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "step", "loss", "eval_loss", "grad_norm", "learning_rate"])  # Define your metrics

    def on_log(self, args, state, control, logs=None, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) == 0 and logs:
            with open(self.csv_file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                epoch = state.epoch
                step = state.global_step
                loss = logs.get("loss", None)
                eval_loss = logs.get("eval_loss", None)
                grad_norm = logs.get("grad_norm", None)  # Adjust this key if necessary
                learning_rate = logs.get("learning_rate", None)

                # Write metrics to the CSV file
                writer.writerow([epoch, step, loss, eval_loss, grad_norm, learning_rate])



def load_checkpoint(trainer, model, resume_from_checkpoint):
    """
    Loads model, optimizer, scheduler, and RNG states from a checkpoint.

    Args:
        trainer (object): Trainer object containing optimizer and scheduler.
        model (torch.nn.Module): The model to load weights into.
        resume_from_checkpoint (str): Path to the checkpoint directory.

    Returns:
        None (Modifies trainer, model, and restores states in-place)
    """
    if not os.path.exists(resume_from_checkpoint):
        raise FileNotFoundError(f"Checkpoint directory {resume_from_checkpoint} does not exist!")

    available_files = os.listdir(resume_from_checkpoint)
    print(f"Files in checkpoint directory: {available_files}")

    adapters_weights = None
    optimizer_state = None
    scheduler_state = None
    trainer_state = None

    # Load Adapter Weights (LoRA)
    if "adapter_model.safetensors" in available_files:
        checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.safetensors")
        print(f"Loading adapter weights from {checkpoint_name}")
        adapters_weights = load_file(checkpoint_name)
    elif "adapter_model.bin" in available_files:
        checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        print(f"Loading adapter weights from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
    else:
        print("No compatible adapter weights found! Ensure 'adapter_model.safetensors' or 'adapter_model.bin' exists.")
        resume_from_checkpoint = None

    # Load Optimizer State
    if "optimizer.pt" in available_files:
        optimizer_checkpoint = os.path.join(resume_from_checkpoint, "optimizer.pt")
        print(f"Loading optimizer state from {optimizer_checkpoint}")
        optimizer_state = torch.load(optimizer_checkpoint)
    else:
        print("No optimizer state found. Optimizer will start fresh.")

    # Load Scheduler State
    if "scheduler.pt" in available_files:
        scheduler_checkpoint = os.path.join(resume_from_checkpoint, "scheduler.pt")
        print(f"Loading scheduler state from {scheduler_checkpoint}")
        scheduler_state = torch.load(scheduler_checkpoint)
    else:
        print("No scheduler state found. Scheduler will start fresh.")

    # Restore Model Adapter Weights (LoRA)
    if adapters_weights is not None:
        try:
            set_peft_model_state_dict(model, adapters_weights)
            print("Adapter weights successfully loaded into the model.")
        except Exception as e:
            raise RuntimeError(f"Failed to set adapter weights: {e}")
    else:
        print("No adapter weights to load. Skipping adapter restoration.")

    # Restore Optimizer & Scheduler
    if trainer.optimizer is not None:
        if optimizer_state is not None:
            try:
                trainer.optimizer.load_state_dict(optimizer_state)
                print("Optimizer state successfully restored.")
            except Exception as e:
                raise RuntimeError(f"Failed to restore optimizer state: {e}")
        else:
            print("Optimizer state not found in checkpoint. Starting with a fresh optimizer.")

    if trainer.lr_scheduler is not None:
        if scheduler_state is not None:
            try:
                trainer.lr_scheduler.load_state_dict(scheduler_state)
                print("Scheduler state successfully restored.")
            except Exception as e:
                raise RuntimeError(f"Failed to restore scheduler state: {e}")
        else:
            print("Scheduler state not found in checkpoint. Starting with a fresh scheduler.")

    # Load Trainer State
    if "trainer_state.json" in available_files:
        trainer_state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
        print(f"Loading trainer state from {trainer_state_path}")
        try:
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
                print("Trainer state successfully loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load trainer state from {trainer_state_path}: {e}")
    else:
        print("Trainer state not found. Training progress may not fully resume.")

    if trainer_state is not None:
        print("Trainer state loaded.")
        current_step = trainer_state.get("step", "Unknown")
        log_history = trainer_state.get("log_history", "None")
        print(f"Current step: {current_step}")
        print(f"Metrics history: {log_history}")
    else:
        print("Trainer state not loaded. Training will resume without previous progress.")
