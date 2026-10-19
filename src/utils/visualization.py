import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def plot_loss_curve(losses, save_path="outputs/loss_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(save_path)
    plt.close()