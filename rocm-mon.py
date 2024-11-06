#!/usr/bin/env python3

"""
This script monitors GPU statistics using the `rocm-smi` command every 5 seconds,
and displays the output along with CPU and RAM usage in a prettified format using the `rich` package.

Usage:
    python rocm-mon.py

Options:
    -h, --help              Show this help message and exit
    -i, --interval          Set the update interval in seconds (default is 5)
    -l, --log-file          Specify a log file to save logs
"""

import argparse
import logging
import subprocess
import sys
import time
from typing import Optional

import psutil
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def show_ascii_art() -> None:
    """
    Display ASCII art banner.
    """
    ascii_art = r"""

██████╗  ██████╗  ██████╗███╗   ███╗
██╔══██╗██╔═══██╗██╔════╝████╗ ████║
██████╔╝██║   ██║██║     ██╔████╔██║
██╔══██╗██║   ██║██║     ██║╚██╔╝██║
██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║
╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝
                                    
   ███╗   ███╗ ██████╗ ███╗   ██╗  
   ████╗ ████║██╔═══██╗████╗  ██║  
   ██╔████╔██║██║   ██║██╔██╗ ██║  
   ██║╚██╔╝██║██║   ██║██║╚██╗██║  
   ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║  
   ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝  
                                                                                                       
    """
    console = Console()
    try:
        console.print(ascii_art, style="bold green")
    except Exception as e:
        logger.error("Failed to display ASCII art: %s", e)


class HelpAction(argparse.Action):
    """
    Custom help action to display ASCII art before showing help message.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super(HelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        show_ascii_art()
        parser.print_help()
        parser.exit()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Monitor GPU statistics using rocm-smi with enhanced display.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='''
Usage Examples:
  python rocm-mon.py
  python rocm-mon.py -i 10
  python rocm-mon.py --interval 10
  python rocm-mon.py -l monitor.log
  python rocm-mon.py -i 10 -l monitor.log
        ''',
    )

    # Override the default help to include ASCII art
    parser.register('action', 'help', HelpAction)

    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=5,
        help='Update interval in seconds.',
    )
    parser.add_argument(
        '-l', '--log-file',
        type=str,
        help='File path to save logs.',
    )
    args = parser.parse_args()

    # Validate interval
    if args.interval <= 0:
        parser.error("Interval must be a positive integer.")

    return args


def setup_logging(log_file: Optional[str]) -> None:
    """
    Setup logging configuration.
    """
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(file_handler)
            logger.info("Logging initialized. Logs will be saved to %s", log_file)
        except (OSError, IOError) as e:
            logger.error("Failed to set up log file '%s': %s", log_file, e)
            sys.exit(1)


def check_rocm_smi_installed() -> None:
    """
    Check if `rocm-smi` is installed and accessible.
    """
    try:
        subprocess.run(['rocm-smi', '--version'], capture_output=True, text=True, check=True)
        logger.debug("rocm-smi is installed.")
    except FileNotFoundError:
        logger.error("`rocm-smi` command not found. Please install ROCm SMI and ensure it's in your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Error checking rocm-smi installation: %s", e)
        sys.exit(1)


def get_rocm_smi_output() -> str:
    """
    Execute the `rocm-smi` command and return its output.
    """
    try:
        result = subprocess.run(['rocm-smi'], capture_output=True, text=True, check=True)
        logger.debug("rocm-smi output:\n%s", result.stdout)
        return result.stdout
    except FileNotFoundError:
        logger.error("`rocm-smi` command not found. Please install ROCm SMI and ensure it's in your PATH.")
        return "Error: rocm-smi command not found."
    except subprocess.CalledProcessError as e:
        logger.error("Failed to execute rocm-smi: %s", e)
        return "Error retrieving GPU information."
    except Exception as e:
        logger.exception("Unexpected error while executing rocm-smi: %s", e)
        return "Error retrieving GPU information."


def get_cpu_ram_usage() -> dict:
    """
    Retrieve current CPU and RAM usage.
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_total = get_size(ram.total)
        ram_used = get_size(ram.used)
        ram_available = get_size(ram.available)
        logger.debug("CPU Usage: %s%%, RAM Usage: %s%% (%s used / %s available)", cpu_percent, ram_percent, ram_used, ram_available)
        return {
            "CPU Usage": f"{cpu_percent}%",
            "RAM Usage": f"{ram_percent}%",
            "RAM Used": ram_used,
            "RAM Available": ram_available,
            "RAM Total": ram_total
        }
    except Exception as e:
        logger.exception("Failed to retrieve CPU and RAM usage: %s", e)
        return {
            "CPU Usage": "N/A",
            "RAM Usage": "N/A",
            "RAM Used": "N/A",
            "RAM Available": "N/A",
            "RAM Total": "N/A"
        }


def get_size(bytes: int, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format.
    """
    try:
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor
        return f"{bytes:.2f}P{suffix}"
    except Exception as e:
        logger.exception("Failed to convert bytes to human-readable format: %s", e)
        return "N/A"


def create_dashboard(gpu_info: str, system_info: dict, interval: int) -> Panel:
    """
    Create a rich Panel containing GPU and system information.
    """
    table = Table(title="System Monitor", box=box.ROUNDED, expand=True)

    # Add GPU Information
    table.add_column("GPU Information", style="cyan", no_wrap=True)
    table.add_row(gpu_info)

    # Add CPU and RAM Usage
    table.add_column("System Usage", style="magenta")
    cpu_ram = (
        f"CPU Usage: {system_info['CPU Usage']}\n"
        f"RAM Usage: {system_info['RAM Usage']} ({system_info['RAM Used']} used / {system_info['RAM Available']} available)"
    )
    table.add_row(cpu_ram)

    panel = Panel(table, title="Monitoring Dashboard", subtitle=f"Updated every {interval} seconds", border_style="bright_blue")
    return panel


def update_dashboard(interval: int) -> Panel:
    """
    Gather all information and create the dashboard panel.
    """
    gpu_info = get_rocm_smi_output()
    system_info = get_cpu_ram_usage()
    dashboard = create_dashboard(gpu_info, system_info, interval)
    return dashboard


def main() -> None:
    """
    Main function to execute the monitoring script.
    """
    try:
        args = parse_arguments()
        setup_logging(args.log_file)
        check_rocm_smi_installed()

        # Always display ASCII art at the start
        show_ascii_art()

        console = Console()
        logger.info("Starting system monitor with an update interval of %d seconds.", args.interval)

        with Live(update_dashboard(args.interval), refresh_per_second=1, console=console) as live:
            while True:
                dashboard = update_dashboard(args.interval)
                live.update(dashboard)
                time.sleep(args.interval)
    except KeyboardInterrupt:
        logger.info("System monitor terminated by user.")
        console = Console()
        console.print("\n[bold red]Monitoring stopped by user.[/bold red]")
    except argparse.ArgumentError as e:
        logger.error("Argument parsing error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        sys.exit(1)


if __name__ == '__main__':
    main()