from torch.optim.lr_scheduler import LambdaLR, PolynomialLR, ExponentialLR, CyclicLR

def custom_schedule_lambda(current_step):
    """
    Custom learning rate schedule function.

    Args:
        current_step (int): The current training step.

    Returns:
        float: The learning rate multiplier.
    """
    if current_step < 585:
        return 1.0  # Full learning rate
    elif 585 <= current_step < 390 * 3:
        return 0.1  # 10% of original LR
    elif 390 * 3 <= current_step < 416 * 6:
        return 0.01  # 1% of original LR
    else:
        return 0.1  # Default multiplier

def get_lr_scheduler(trainer, learning_rate_policy, option):
    """
    Assigns the appropriate learning rate scheduler to the trainer based on the given policy.

    Args:
        trainer (object): The trainer object containing the optimizer.
        learning_rate_policy (str): The chosen learning rate schedule policy.
        option (str): The option specifying the schedule parameters.

    Returns:
        None (Modifies trainer.lr_scheduler in-place)
    """

    if learning_rate_policy == "stepLR":
        print("policy StepLR")
        if option == "custom":  # Updated from "A" to "custom"
            print("Using StepLR with custom phase adjustments")
            trainer.lr_scheduler = LambdaLR(
                trainer.optimizer, 
                lr_lambda=custom_schedule_lambda  # Now calling the external function
            )

    elif learning_rate_policy == "poly":
        print("policy Polynomial Decay")
        total_epochs = 5
        total_iters = total_epochs * 390  # Total iterations for full training

        poly_powers = {
            "zero_point_one": 0.1,
            "zero_point_three": 0.3,
            "zero_point_five": 0.5,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5
        }

        if option in poly_powers:
            power = poly_powers[option]
            trainer.lr_scheduler = PolynomialLR(
                trainer.optimizer,
                total_iters=total_iters ,  
                power=power
            )
        else:
            print("ERROR: No valid option selected") 

    elif learning_rate_policy == "lambda":
        print("policy Lambda N Step Decay")
        trainer.lr_scheduler = LambdaLR(
            trainer.optimizer,
            lr_lambda=custom_schedule_lambda  # Using the externally defined lambda
        )

    elif learning_rate_policy == "exp":
        print("policy Exponential Decay")
        exp_rates = {
            "point_nine_eight": 0.98,
            "point_nine_nine": 0.99,
            "point_nine_nine_five": 0.995
        }

        if option in exp_rates:
            gamma = exp_rates[option]
            trainer.lr_scheduler = ExponentialLR(
                trainer.optimizer,
                gamma=gamma
            )
        else:
            print("ERROR: No valid option selected")

    elif learning_rate_policy == "cyclic":
        print("policy Cyclic LR with half range")
        gamma_values  = {
            "NineNineNine": 0.999,
            "NineNineEight": 0.998,
            "NineNineSeven": 0.997,
        }

        step_size = 325
        base_lr = 1e-6
        max_lr = 1e-3

        if option in gamma_values:
            gamma = gamma_values[option]
            trainer.lr_scheduler = CyclicLR(
                trainer.optimizer,
                base_lr=base_lr,
                max_lr=max_lr,
                step_size_up=step_size,
                step_size_down=step_size,
                mode='exp_range',
                gamma=gamma,
            )
        else:
            print("ERROR: No valid cyclic LR option selected")

    elif learning_rate_policy == "constant":
        print("policy Constant")
    else:
        raise ValueError(f"Unsupported learning_rate_policy: {learning_rate_policy}")

    # Print optimizer and scheduler for verification
    print("Scheduler Set:", trainer.lr_scheduler)
