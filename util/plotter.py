import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import Slider, Button


class PlotterConfigs:
    def __init__(self):
        self.plot_train_loss = True
        self.plot_val_loss = True
        self.plot_grad_norm = True
        self.plot_metric_losses = True
        self.smooth_sigma = 2
        self.figsize = (10, 5)
        self.dpi = 300
        self.output_dir = 'plots'
        self.controller = 'Ant-v1'
        self.eval_period = 50
        self.interactive = False


class Plotter:
    def __init__(self, config: PlotterConfigs):
        self.config = config
        self.metric_colors = plt.cm.jet(np.linspace(0, 1, 10))

    def gaussian_kernel(self, size, sigma):
        size = int(size) // 2
        x = np.arange(-size, size + 1, dtype=np.float32)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        return kernel

    def smooth_curve(self, points, sigma):
        kernel_size = int(4 * sigma + 1)
        if len(points) < kernel_size:
            return points
        kernel = self.gaussian_kernel(kernel_size, sigma)
        return np.convolve(points, kernel, mode='same')

    def plot_losses(self, losses, title, label, color, linestyle='-'):
        if self.config.eval_period:
            x_values = [i * self.config.eval_period for i in range(len(losses))]
        else:
            x_values = list(range(len(losses)))
        plt.plot(
            x_values,
            self.smooth_curve(losses, self.config.smooth_sigma),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
        plt.xlabel('Steps', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title(title, fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)

    def plot_metrics(self, metric_losses, title):
        for i, (metric, losses) in enumerate(metric_losses.items()):
            color = self.metric_colors[i % len(self.metric_colors)]
            self.plot_losses(losses, title, metric, color)

    def plot_grad_norm(self, grad_norms, title):
        self.plot_losses(
            grad_norms, title, 'Gradient Norm', 'purple', linestyle='--'
        )

    def plot_interactive(
        self, train_losses, val_losses, metric_losses, grad_norms
    ):
        fig, ax = plt.subplots(figsize=self.config.figsize)
        plt.subplots_adjust(bottom=0.25)

        # Plot initial losses and metrics
        self.plot_losses(train_losses, '', 'Training Loss', 'blue')
        self.plot_losses(
            val_losses, '', 'Validation Loss', 'orange', linestyle='--'
        )
        self.plot_metrics(metric_losses, '')
        self.plot_grad_norm(grad_norms, '')

        # Add sliders for adjusting the smoothing and x-axis range
        smooth_slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
        smooth_slider = Slider(
            smooth_slider_ax,
            'Smoothing',
            0.1,
            10.0,
            valinit=self.config.smooth_sigma,
            valstep=0.1,
        )

        range_slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
        range_slider = Slider(
            range_slider_ax, 'Range', 0.1, 1.0, valinit=1.0, valstep=0.1
        )

        def update(val):
            ax.clear()
            self.config.smooth_sigma = smooth_slider.val
            range_factor = range_slider.val

            n_train = int(len(train_losses) * range_factor)
            n_val = int(len(val_losses) * range_factor)
            n_metric = {
                k: int(len(v) * range_factor) for k, v in metric_losses.items()
            }
            n_grad = int(len(grad_norms) * range_factor)

            self.plot_losses(
                train_losses[:n_train], '', 'Training Loss', 'blue'
            )
            self.plot_losses(
                val_losses[:n_val],
                '',
                'Validation Loss',
                'orange',
                linestyle='--',
            )
            self.plot_metrics(
                {k: v[: n_metric[k]] for k, v in metric_losses.items()}, ''
            )
            self.plot_grad_norm(grad_norms[:n_grad], '')
            fig.canvas.draw_idle()

        smooth_slider.on_changed(update)
        range_slider.on_changed(update)

        # Add a button for saving the plot
        save_button_ax = plt.axes([0.8, 0.01, 0.1, 0.04])
        save_button = Button(save_button_ax, 'Save', hovercolor='0.975')

        def save_plot(event):
            output_path = os.path.join(
                self.config.output_dir,
                f'{self.config.controller}_interactive.png',
            )
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            print(f'Interactive plot saved as {output_path}')

        save_button.on_clicked(save_plot)

        plt.show()

    def plot(
        self,
        train_losses: list[float],
        val_losses: list[float],
        metric_losses: dict[str, list[float]],
        grad_norms: list[float],
    ):
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)

        if self.config.interactive:
            self.plot_interactive(
                train_losses, val_losses, metric_losses, grad_norms
            )
        else:
            # Plot training and validation losses
            if self.config.plot_train_loss and self.config.plot_val_loss:
                plt.figure(figsize=self.config.figsize)
                self.plot_losses(
                    train_losses,
                    f'{self.config.controller} Training Loss',
                    'Training Loss',
                    'blue',
                )
                self.plot_losses(
                    val_losses, '', 'Validation Loss', 'orange', linestyle='--'
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.config.output_dir,
                        f'{self.config.controller}_losses.png',
                    ),
                    dpi=self.config.dpi,
                )
                print(f'Saved at {os.path.join(self.config.output_dir, f"{self.config.controller}_losses.png")}')
                plt.close()

            # Plot metric losses
            if self.config.plot_metric_losses and metric_losses:
                plt.figure(figsize=self.config.figsize)
                self.plot_metrics(
                    metric_losses, f'{self.config.controller} Metric Losses'
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.config.output_dir,
                        f'{self.config.controller}_metrics.png',
                    ),
                    dpi=self.config.dpi,
                )
                plt.close()

            # Plot gradient norms
            if self.config.plot_grad_norm and len(grad_norms) > 0:
                plt.figure(figsize=self.config.figsize)
                self.plot_grad_norm(
                    grad_norms, f'{self.config.controller} Gradient Norms'
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        self.config.output_dir,
                        f'{self.config.controller}_grad_norms.png',
                    ),
                    dpi=self.config.dpi,
                )
                plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics')
    parser.add_argument(
        '--train_losses',
        type=str,
        default='plots/transformer_train_losses.npy',
        help='Path to the training losses file',
    )
    parser.add_argument(
        '--val_losses',
        type=str,
        default='plots/transformer_val_losses.npy',
        help='Path to the validation losses file',
    )
    parser.add_argument(
        '--metric_losses',
        type=str,
        nargs='+',
        default=[
            'plots/transformer_coordinate_loss.npy',
            'plots/transformer_orientation_loss.npy',
            'plots/transformer_angle_loss.npy',
            'plots/transformer_coordinate_velocity_loss.npy',
            'plots/transformer_angular_velocity_loss.npy',
        ],
        help='Paths to the metric losses files',
    )
    parser.add_argument(
        '--grad_norms',
        type=str,
        default='plots/transformer_grad_norms.npy',
        help='Path to the gradient norms file',
    )
    args = parser.parse_args()

    config = PlotterConfigs()
    plotter = Plotter(config)

    train_losses = np.load(args.train_losses)
    val_losses = np.load(args.val_losses)

    metric_names = [
        'Coordinate Loss',
        'Orientation Loss',
        'Angle Loss',
        'Coordinate Velocity Loss',
        'Angular Velocity Loss',
    ]
    metric_losses = {
        name: np.load(path)
        for name, path in zip(metric_names, args.metric_losses, strict=True)
    }

    grad_norms = np.load(args.grad_norms)

    plotter.plot(train_losses, val_losses, metric_losses, grad_norms)

if __name__ == '__main__':
    main()