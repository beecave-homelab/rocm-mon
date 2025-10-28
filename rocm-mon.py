#!venv/bin/python3
# #!/usr/bin/env python3

"""ROCm System Monitor.

This script monitors GPU statistics using the `rocm-smi` command every 5 seconds,
and displays the output along with CPU and RAM usage in a prettified format using
the `rich` package.

Usage:
    venv/bin/python rocm-mon.py [OPTIONS]

Options:
    -i, --interval INTEGER  Update interval in seconds (default: 5)
    -l, --log-file TEXT    File path to save logs
    --help                 Show this message and exit.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping

import click
import psutil
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Constants
DEFAULT_INTERVAL: int = 5
DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
BYTE_UNITS: list[str] = ["", "K", "M", "G", "T", "P"]
BYTES_FACTOR: int = 1024
SEPARATOR_LINE: str = "─" * 50

# Color constants
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"

# Thresholds for color coding
CPU_THRESHOLDS = {"low": 50, "medium": 80}  # percentage
RAM_THRESHOLDS = {"low": 60, "medium": 80}  # percentage
SWAP_THRESHOLDS = {"low": 30, "medium": 60}  # percentage


def get_color_for_value(value: float, thresholds: dict[str, float]) -> str:
    """Return ANSI color code based on value and thresholds.

    Returns:
        str: ANSI color code string for the provided value and thresholds.

    """
    try:
        value_float = float(value.strip("%"))
        if value_float <= thresholds["low"]:
            return GREEN
        elif value_float <= thresholds["medium"]:
            return YELLOW
        return RED
    except (ValueError, AttributeError):
        return ""


def get_system_status(system_info: Mapping[str, str]) -> tuple[str, str]:
    """Generate a status message and color based on system metrics.

    Returns:
        tuple[str, str]: A tuple of (status_message, ansi_color_code).

    """
    try:
        cpu_val = float(system_info["CPU Usage"].split("%")[0])
        ram_val = float(system_info["RAM Usage"].strip("%"))
        status_color = GREEN

        if cpu_val > CPU_THRESHOLDS["medium"] or ram_val > RAM_THRESHOLDS["medium"]:
            status_color = RED
            return "System under heavy load! Check resource usage.", status_color
        elif cpu_val > CPU_THRESHOLDS["low"] or ram_val > RAM_THRESHOLDS["low"]:
            status_color = YELLOW
            return "System experiencing moderate load.", status_color
        return "System running normally. All metrics within safe limits.", status_color
    except (ValueError, KeyError):
        return "Unable to determine system status.", ""


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


class SystemMonitor:
    """Handle system monitoring operations."""

    def __init__(self, interval: int = DEFAULT_INTERVAL) -> None:
        """Initialize the system monitor.

        Args:
            interval: Update interval in seconds

        """
        self.interval = interval
        self.console = Console()

    @staticmethod
    def get_size(bytes_value: int, suffix: str = "B") -> str:
        """Scale bytes to its proper format.

        Args:
            bytes_value: Number of bytes to format
            suffix: Suffix to append to the unit (default: "B")

        Returns:
            str: Formatted string representing the size with appropriate unit

        """
        try:
            for unit in BYTE_UNITS:
                if bytes_value < BYTES_FACTOR:
                    return f"{bytes_value:.2f}{unit}{suffix}"
                bytes_value /= BYTES_FACTOR
            return f"{bytes_value:.2f}P{suffix}"
        except Exception as e:
            logger.exception("Failed to convert bytes to human-readable format: %s", e)
            return "N/A"

    def get_cpu_ram_usage(self) -> Mapping[str, str]:
        """Retrieve current CPU and RAM usage statistics.

        Returns:
            Mapping[str, str]: Dictionary containing CPU and RAM usage information

        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            cpu_freq_current = f"{cpu_freq.current:.1f}MHz" if cpu_freq else "N/A"

            try:
                ram = psutil.virtual_memory()
                swap = psutil.swap_memory()
                ram_info = {
                    "CPU Usage": f"{cpu_percent}% @ {cpu_freq_current}",
                    "RAM Usage": f"{ram.percent}%",
                    "RAM Used": self.get_size(ram.used),
                    "RAM Available": self.get_size(ram.available),
                    "RAM Total": self.get_size(ram.total),
                    "Swap Used": f"{self.get_size(swap.used)} ({swap.percent}%)",
                }
            except AttributeError:
                ram = psutil.virtual_memory()
                ram_info = {
                    "CPU Usage": f"{cpu_percent}%",
                    "RAM Usage": f"{ram.percent}%",
                    "RAM Used": self.get_size(getattr(ram, "used", 0)),
                    "RAM Available": self.get_size(getattr(ram, "available", 0)),
                    "RAM Total": self.get_size(getattr(ram, "total", 0)),
                    "Swap Used": "N/A",
                }

            logger.debug(
                "CPU: %s, RAM: %s (%s used / %s available)",
                ram_info["CPU Usage"],
                ram_info["RAM Usage"],
                ram_info["RAM Used"],
                ram_info["RAM Available"],
            )

            return ram_info

        except Exception as e:
            logger.exception("Failed to retrieve CPU and RAM usage: %s", e)
            return {
                "CPU Usage": "N/A",
                "RAM Usage": "N/A",
                "RAM Used": "N/A",
                "RAM Available": "N/A",
                "RAM Total": "N/A",
                "Swap Used": "N/A",
            }

    def create_dashboard(self, gpu_info: str, system_info: Mapping[str, str]) -> Panel:
        """Create a rich Panel containing system information.

        Args:
            gpu_info: String containing GPU information from rocm-smi
            system_info: Dictionary containing CPU and RAM usage information

        Returns:
            Panel: Rich panel object containing the formatted dashboard

        """
        # Create the main table
        table = Table(box=box.ROUNDED, expand=True, show_header=False)
        table.add_column("Content", justify="left", style="white")

        # Format CPU usage with dynamic color
        cpu_usage = system_info["CPU Usage"].split("@")[0].strip()
        cpu_freq = (
            system_info["CPU Usage"].split("@")[1].strip()
            if "@" in system_info["CPU Usage"]
            else ""
        )
        cpu_val = float(cpu_usage.strip("%"))
        cpu_style = "green" if cpu_val < 50 else "yellow" if cpu_val < 80 else "red"

        # Format RAM usage with dynamic color
        ram_val = float(system_info["RAM Usage"].strip("%"))
        ram_style = "green" if ram_val < 60 else "yellow" if ram_val < 80 else "red"
        ram_text = (
            f"{system_info['RAM Usage']} "
            f"({system_info['RAM Used']}/{system_info['RAM Total']})"
        )

        # Format Swap usage with dynamic color
        swap_usage = system_info["Swap Used"].split("(")[-1].strip(")")
        swap_val = float(swap_usage.strip("%"))
        swap_style = "green" if swap_val < 30 else "yellow" if swap_val < 60 else "red"

        # Create the system stats line with rich Text objects
        system_stats = Text()
        system_stats.append("CPU: ", style="bold")
        system_stats.append(f"{cpu_usage} @ {cpu_freq}", style=cpu_style)
        system_stats.append(" | ", style="white")
        system_stats.append("RAM: ", style="bold")
        system_stats.append(ram_text, style=ram_style)
        system_stats.append(" | ", style="white")
        system_stats.append("Swap: ", style="bold")
        system_stats.append(swap_usage, style=swap_style)

        # Get system status with emoji
        status_msg, status_color = get_system_status(system_info)
        status_emoji = (
            "✅ "
            if status_color == GREEN
            else "⚠️ "
            if status_color == YELLOW
            else "🔴 "
        )
        status_text = Text()
        status_text.append(status_emoji)
        style_plain = status_color.replace("\u001b", "")
        status_text.append(status_msg, style=style_plain)

        # Parse GPU info and create GPU panel
        gpu_parts = gpu_info.split("│")
        if len(gpu_parts) >= 3:
            temp = gpu_parts[1].strip()
            vram = gpu_parts[2].strip()
            usage = gpu_parts[3].strip() if len(gpu_parts) > 3 else "N/A"

            gpu_text = Text()
            gpu_text.append("Temperature: ", style="bold cyan")
            gpu_text.append(temp)
            gpu_text.append(" | ", style="white")
            gpu_text.append("VRAM: ", style="bold cyan")
            gpu_text.append(vram)
            gpu_text.append(" | ", style="white")
            gpu_text.append("Usage: ", style="bold cyan")
            gpu_text.append(usage)
        else:
            gpu_text = Text(gpu_info)

        # Add content to table
        table.add_row("")  # Spacing
        table.add_row(
            Panel(system_stats, title="System Information", border_style="cyan")
        )
        table.add_row(status_text)
        table.add_row("")  # Spacing
        table.add_row(Panel(gpu_text, title="GPU Information", border_style="magenta"))
        table.add_row("")  # Spacing

        # Create footer with update time
        current_time = time.strftime("%H:%M:%S")
        footer = Text()
        footer.append("Last Update: ", style="dim")
        footer.append(current_time, style="bright_black")
        footer.append(" | ", style="dim")
        footer.append(
            f"Refresh Interval: {self.interval} seconds", style="bright_black"
        )
        table.add_row(footer)

        return Panel(
            table,
            title="[bold blue]ROCM MON[/bold blue]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def update_dashboard(self) -> Panel:
        """Gather all information and create the dashboard panel.

        Returns:
            Panel: The rendered dashboard panel ready for display.

        """
        gpu_info = get_rocm_smi_output()
        system_info = self.get_cpu_ram_usage()
        return self.create_dashboard(gpu_info, system_info)

    def run(self) -> None:
        """Run the system monitor."""
        try:
            with Live(
                self.update_dashboard(), refresh_per_second=1, console=self.console
            ) as live:
                while True:
                    dashboard = self.update_dashboard()
                    live.update(dashboard)
                    time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("System monitor terminated by user.")
            self.console.print("\n[bold red]Monitoring stopped by user.[/bold red]")


def show_ascii_art() -> None:
    """Display ASCII art banner."""
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


def check_rocm_smi_installed() -> None:
    """Check if rocm-smi is installed and accessible."""
    try:
        subprocess.run(
            ["rocm-smi", "--version"], capture_output=True, text=True, check=True
        )
        logger.debug("rocm-smi is installed.")
    except FileNotFoundError:
        logger.error(
            "`rocm-smi` command not found. "
            "Please install ROCm SMI and ensure it's in your PATH."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error("Error checking rocm-smi installation: %s", e)
        sys.exit(1)


def setup_logging(log_file: str | None) -> None:
    """Set up logging configuration.

    Args:
        log_file: Optional path to log file

    """
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
            logger.addHandler(file_handler)
            logger.info("Logging initialized. Logs will be saved to %s", log_file)
        except OSError as e:
            logger.error("Failed to set up log file '%s': %s", log_file, e)
            sys.exit(1)


def get_rocm_smi_output() -> str:
    """Execute the rocm-smi command and extract GPU stats.

    Returns:
        str: Formatted string containing GPU usage, VRAM usage, and temperature

    """
    try:
        result = subprocess.run(
            ["rocm-smi"], capture_output=True, text=True, check=True
        )

        # Parse the output to extract relevant information
        lines = result.stdout.strip().split("\n")
        stats = None

        # Find the line with actual GPU stats (contains % and °C)
        for line in lines:
            if "°C" in line and "%" in line:
                stats = line.strip()
                break

        if not stats:
            return "Error: Could not find GPU statistics in rocm-smi output"

        # Extract values using string splitting and indexing
        # The line format is consistent, so we can split by whitespace
        parts = [p for p in stats.split() if p]

        # Extract temperature (format: XX.X°C)
        temp = next((p for p in parts if "°C" in p), "N/A")
        # Extract VRAM% (format: XX%)
        vram = next((p for p in parts if "%" in p), "N/A")
        # Extract GPU% (last percentage in the line)
        gpu = parts[-1] if parts else "N/A"

        # Format the output
        return (
            f"GPU Stats │ Temperature: {temp} │ VRAM Usage: {vram} │ GPU Usage: {gpu}"
        )

    except FileNotFoundError:
        logger.error(
            "`rocm-smi` command not found. "
            "Please install ROCm SMI and ensure it's in your PATH."
        )
        return "Error: rocm-smi command not found."
    except subprocess.CalledProcessError as e:
        logger.error("Failed to execute rocm-smi: %s", e)
        return "Error retrieving GPU information."
    except Exception as e:
        logger.exception("Unexpected error while executing rocm-smi: %s", e)
        return "Error retrieving GPU information."


def check_system_compatibility() -> None:
    """Check if the system supports CPU and RAM monitoring via psutil."""
    try:
        if sys.platform.startswith("linux"):
            if not os.path.exists("/proc/stat") or not os.access("/proc/stat", os.R_OK):
                logger.error("Cannot access /proc/stat. CPU monitoring may be limited.")
            if not os.path.exists("/proc/meminfo") or not os.access(
                "/proc/meminfo", os.R_OK
            ):
                logger.error(
                    "Cannot access /proc/meminfo. RAM monitoring may be limited."
                )

        psutil.cpu_percent(interval=0.1)
        psutil.virtual_memory()
        logger.debug("System compatibility check passed.")
    except (PermissionError, OSError) as e:
        logger.error("System monitoring may be limited due to permissions: %s", e)
    except Exception as e:
        logger.error("Unexpected error during system compatibility check: %s", e)


@click.command(help="Monitor GPU statistics using rocm-smi with enhanced display.")
@click.option(
    "-i",
    "--interval",
    default=DEFAULT_INTERVAL,
    help="Update interval in seconds.",
    type=int,
)
@click.option("-l", "--log-file", help="File path to save logs.", type=str)
def main(interval: int, log_file: str | None) -> None:
    """Run the system monitoring application.

    Args:
        interval: Update interval in seconds
        log_file: Optional path to log file

    Raises:
        click.BadParameter: If the provided interval is not a positive integer.

    """
    try:
        if interval <= 0:
            raise click.BadParameter("Interval must be a positive integer.")

        setup_logging(log_file)
        check_rocm_smi_installed()
        check_system_compatibility()
        show_ascii_art()

        monitor = SystemMonitor(interval)
        monitor.run()

    except click.BadParameter as e:
        logger.error("Parameter error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
