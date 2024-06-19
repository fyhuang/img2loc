import torch

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str)
    parser.add_argument('out_file', type=str)

    mode_args = []
    parser.add_argument('--reset_lr', type=float, default=None)
    mode_args.append("reset_lr")

    args = parser.parse_args()

    # Check that only one of the mode arguments is set
    num_set = sum(1 for m in mode_args if getattr(args, m) is not None)
    if num_set != 1:
        print(f"Exactly one of the following arguments must be set: {mode_args}")
        sys.exit(1)

    checkpoint = torch.load(args.in_file)

    if args.reset_lr is not None:
        checkpoint["lr_schedulers"][0]["best"] = 0.0
        checkpoint["lr_schedulers"][0]["_last_lr"] = [args.reset_lr]
        checkpoint['optimizer_states'][0]['param_groups'][0]['lr'] = args.reset_lr
    
    torch.save(checkpoint, args.out_file)

if __name__ == "__main__":
    main()