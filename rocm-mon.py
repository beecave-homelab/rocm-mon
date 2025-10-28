#!/usr/bin/env python3

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

import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping

import click
import psutil
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Constants
DEFAULT_INTERVAL: int = 5
DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(levelname)s - %(message)s"
BYTE_UNITS: list[str] = ["", "K", "M", "G", "T", "P"]
BYTES_FACTOR: int = 1024
SEPARATOR_LINE: str = "─" * 50

THEME = Theme(
    {
        "title": "bold bright_blue",
        "muted": "bright_black",
        "ok": "green",
        "warn": "yellow",
        "alert": "red",
        "sys.border": "cyan",
        "gpu.border": "magenta",
    }
)

# Thresholds for color coding
CPU_THRESHOLDS = {"low": 50, "medium": 80}  # percentage
RAM_THRESHOLDS = {"low": 60, "medium": 80}  # percentage
SWAP_THRESHOLDS = {"low": 30, "medium": 60}  # percentage
TEMP_THRESHOLDS = {"low": 70, "medium": 85}  # Celsius


def style_for_percent(value: float, thresholds: dict[str, float]) -> str:
    """Return a Rich style name based on ``value`` and percentage thresholds.

    Args:
        value: Numeric value (0-100 range expected for percentages).
        thresholds: Mapping with "low" and "medium" cutoffs.

    Returns:
        A style name: "ok", "warn", or "alert".

    """
    try:
        v = float(str(value).strip("%"))
    except (ValueError, AttributeError):
        return "muted"
    if v <= thresholds["low"]:
        return "ok"
    if v <= thresholds["medium"]:
        return "warn"
    return "alert"


def get_system_status(system_info: Mapping[str, str]) -> tuple[str, str]:
    """Generate a status message and color based on system metrics.

    Returns:
        tuple[str, str]: A tuple of (status_message, ansi_color_code).

    """
    try:
        cpu_val = float(system_info["CPU Usage"].split("%")[0])
        ram_val = float(system_info["RAM Usage"].strip("%"))
        style = "ok"
        if cpu_val > CPU_THRESHOLDS["medium"] or ram_val > RAM_THRESHOLDS["medium"]:
            style = "alert"
            return "System under heavy load! Check resource usage.", style
        if cpu_val > CPU_THRESHOLDS["low"] or ram_val > RAM_THRESHOLDS["low"]:
            style = "warn"
            return "System experiencing moderate load.", style
        return "System running normally. All metrics within safe limits.", style
    except (ValueError, KeyError):
        return "Unable to determine system status.", "muted"


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
handler.setFormatter(formatter)
logger.addHandler(handler)


class SystemMonitor:
    """Handle system monitoring operations."""

    def __init__(
        self,
        interval: int = DEFAULT_INTERVAL,
        *,
        compact: bool = False,
        no_ascii: bool = False,
        console: Console | None = None,
    ) -> None:
        """Initialize the system monitor.

        Args:
            interval: Update interval in seconds
            compact: Render compact single-line/device layout
            no_ascii: Disable ASCII banner
            console: Optional Console (to allow no-color or themed consoles)

        """
        self.interval = interval
        self.compact = compact
        self.no_ascii = no_ascii
        self.console = console or Console(theme=THEME)

    @staticmethod
    def _progress(label: str, percent: float, style: str) -> Progress:
        """Create a horizontal gauge for ``percent`` with a label.

        Args:
            label: Metric label.
            percent: Percentage value 0-100.
            style: Rich style name for the gauge color.

        Returns:
            A configured Progress renderable with a single task.

        """
        prog = Progress(
            TextColumn(f"[bold]{label}[/]"),
            BarColumn(bar_width=None, style=style, complete_style=style),
            TextColumn("{task.percentage:>4.0f}%"),
            expand=True,
        )
        prog.add_task("", total=100, completed=max(0, min(100, int(percent))))
        return prog

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

    def create_dashboard(
        self,
        gpu_info: Mapping[str, str],
        system_info: Mapping[str, str],
    ) -> Panel:
        """Create a rich Panel containing system information.

        Args:
            gpu_info: String containing GPU information from rocm-smi
            system_info: Dictionary containing CPU and RAM usage information

        Returns:
            Panel: Rich panel object containing the formatted dashboard

        """
        # Header (include ASCII art inside header panel if enabled)
        header_parts: list[Text] = []
        if not self.no_ascii:
            # Render only ASCII banner when enabled; center it in the header
            header_parts.append(
                Text(show_ascii_art(), style="bold green", justify="center")
            )
        else:
            # Fallback: centered text title when ASCII is disabled
            header_parts.append(Text("ROCM MON", style="title", justify="center"))
        header_group = Group(*header_parts)

        # System block
        cpu_usage_str = system_info["CPU Usage"].split("@")[0].strip()
        cpu_freq = (
            system_info["CPU Usage"].split("@")[1].strip()
            if "@" in system_info["CPU Usage"]
            else ""
        )
        cpu_val = float(cpu_usage_str.strip("%"))
        cpu_prog = self._progress(
            "CPU",
            cpu_val,
            style_for_percent(cpu_val, CPU_THRESHOLDS),
        )

        ram_val = float(system_info["RAM Usage"].strip("%"))
        ram_prog = self._progress(
            f"RAM ({system_info['RAM Used']}/{system_info['RAM Total']})",
            ram_val,
            style_for_percent(ram_val, RAM_THRESHOLDS),
        )

        swap_pct_text = system_info["Swap Used"].split("(")[-1].strip(")")
        swap_val = (
            float(swap_pct_text.strip("%"))
            if swap_pct_text.endswith("%")
            else float(swap_pct_text)
        )
        swap_prog = self._progress(
            "Swap",
            swap_val,
            style_for_percent(swap_val, SWAP_THRESHOLDS),
        )

        sys_table = Table.grid(padding=(0, 1))
        sys_table.add_row(cpu_prog)
        if cpu_freq:
            sys_table.add_row(Text(f"Frequency: {cpu_freq}", style="muted"))
        sys_table.add_row(ram_prog)
        sys_table.add_row(swap_prog)

        status_msg, status_style = get_system_status(system_info)
        status_badge = Text.assemble((" ",), (status_msg, status_style))
        sys_panel = Panel.fit(
            Group(sys_table, status_badge),
            title="System",
            border_style="sys.border",
            box=box.ROUNDED,
            padding=(1, 1),
        )

        # GPU block
        if gpu_info.get("error"):
            gpu_content = Text(gpu_info["error"], style="alert")
        else:
            gpu_prog = self._progress(
                "GPU",
                float(gpu_info.get("gpu_percent", 0.0)),
                style_for_percent(
                    float(gpu_info.get("gpu_percent", 0.0)), CPU_THRESHOLDS
                ),
            )
            vram_prog = self._progress(
                (
                    "VRAM ("
                    f"{gpu_info.get('vram_used', 'N/A')}/"
                    f"{gpu_info.get('vram_total', 'N/A')}"
                    ")"
                ),
                float(gpu_info.get("vram_percent", 0.0)),
                style_for_percent(
                    float(gpu_info.get("vram_percent", 0.0)), RAM_THRESHOLDS
                ),
            )
            temp_c = float(gpu_info.get("temp_c", 0.0))
            temp_pct = min(100.0, (temp_c / TEMP_THRESHOLDS["medium"]) * 100.0)
            temp_prog = self._progress(
                "Temp",
                temp_pct,
                style_for_percent(temp_c, TEMP_THRESHOLDS),
            )
            temp_text = Text(
                f"{temp_c:.1f}°C",
                style=style_for_percent(temp_c, TEMP_THRESHOLDS),
            )

            meta = Text(
                gpu_info.get("name", "GPU"),
                style="muted",
            )
            gtable = Table.grid(padding=(0, 1))
            gtable.add_row(meta)
            gtable.add_row(gpu_prog)
            gtable.add_row(vram_prog)
            gtable.add_row(Group(temp_prog, temp_text))
            # Optional: show per-process info if available
            procs = gpu_info.get("processes") or []
            if procs:
                ptable = Table.grid(padding=(0, 1))
                # Header row
                header = Text("PID    NAME           GPU  VRAM", style="muted")
                ptable.add_row(header)
                # Show up to 5 processes to keep the panel compact
                for proc in procs[:5]:
                    pid = str(proc.get("pid", "-"))
                    name = str(proc.get("name", "-"))[:14]
                    gpus = str(proc.get("gpus", "-"))
                    vram = str(proc.get("vram_used", "-"))
                    row = Text(f"{pid:>5}  {name:<14}  {gpus:>3}  {vram:>6}")
                    ptable.add_row(row)
                gtable.add_row(Text("Processes", style="bold"))
                gtable.add_row(ptable)
            gpu_content = gtable

        gpu_panel = Panel.fit(
            gpu_content,
            title="GPU",
            border_style="gpu.border",
            box=box.ROUNDED,
            padding=(1, 1),
        )

        # Layout
        layout = Layout()
        # Determine a compact header height (ASCII lines + title line)
        ascii_lines = 0 if self.no_ascii else len(show_ascii_art().splitlines())
        # Add extra breathing room (+3) so glyphs never clip against the border
        header_lines = ascii_lines + 3  # +1 title +2 top/bottom space
        header_size = max(6, min(10, header_lines))
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )
        # Body as a two-column Table to keep minimal height and equal widths
        body = Table.grid(expand=True)
        body.add_column(ratio=1)
        body.add_column(ratio=1)
        body.add_row(sys_panel, gpu_panel)
        layout["header"].update(
            Panel(
                header_group,
                border_style="bright_blue",
                box=box.SQUARE,
                padding=(0, 1),
            )
        )
        layout["body"].update(body)

        # Footer
        current_time = time.strftime("%H:%M:%S")
        footer = Text.assemble(
            ("Last Update: ", "muted"),
            (current_time, "muted"),
            (" | ", "muted"),
            (f"Refresh Interval: {self.interval} seconds", "muted"),
        )
        layout["footer"].update(footer)

        return Panel(layout, box=box.SIMPLE)

    def update_dashboard(self) -> Panel:
        """Gather all information and create the dashboard panel.

        Returns:
            Panel: The rendered dashboard panel ready for display.

        """
        gpu_info = get_rocm_smi_output()
        system_info = self.get_cpu_ram_usage()
        return self.create_dashboard(
            gpu_info,
            system_info,
        )

    def run(self) -> None:
        """Run the system monitor."""
        try:
            if self.console.is_terminal and not self.compact:
                with Live(
                    self.update_dashboard(),
                    refresh_per_second=1,
                    console=self.console,
                ) as live:
                    while True:
                        dashboard = self.update_dashboard()
                        live.update(dashboard)
                        if getattr(self, "_once", False):
                            break
                        time.sleep(self.interval)
            else:
                # Compact, single render (TTY-small or non-TTY)
                sys_info = self.get_cpu_ram_usage()
                gpu = get_rocm_smi_output()
                self.console.print(self.render_compact_line(sys_info, gpu))
        except KeyboardInterrupt:
            logger.info("System monitor terminated by user.")
            self.console.print("\n[bold red]Monitoring stopped by user.[/bold red]")

    def render_compact_line(
        self, system_info: Mapping[str, str], gpu_info: Mapping[str, str]
    ) -> Text:
        """Render a compact, single-line status suitable for small or non-TTY outputs.

        Args:
            system_info: CPU/RAM/Swap metrics.
            gpu_info: Parsed GPU metrics mapping.

        Returns:
            A Rich Text object representing one concise status line.

        """
        cpu_pct = float(system_info["CPU Usage"].split("%")[0])
        ram_pct = float(system_info["RAM Usage"].strip("%"))
        swap_pct = float(system_info["Swap Used"].split("(")[-1].strip(")%"))
        text = Text()
        text.append("CPU ", style="bold")
        text.append(
            f"{cpu_pct:.0f}% ",
            style=style_for_percent(cpu_pct, CPU_THRESHOLDS),
        )
        text.append("RAM ", style="bold")
        text.append(
            f"{ram_pct:.0f}% ({system_info['RAM Used']}/{system_info['RAM Total']}) ",
            style=style_for_percent(ram_pct, RAM_THRESHOLDS),
        )
        text.append("SWAP ", style="bold")
        text.append(
            f"{swap_pct:.0f}% ", style=style_for_percent(swap_pct, SWAP_THRESHOLDS)
        )

        if gpu_info.get("error"):
            text.append("GPU N/A", style="alert")
            return text

        gpu_pct = float(gpu_info.get("gpu_percent", 0.0))
        vram_pct = float(gpu_info.get("vram_percent", 0.0))
        temp_c = float(gpu_info.get("temp_c", 0.0))
        text.append("| GPU ", style="bold")
        text.append(
            f"{gpu_pct:.0f}% ",
            style=style_for_percent(gpu_pct, CPU_THRESHOLDS),
        )
        text.append("VRAM ", style="bold")
        text.append(
            (
                f"{vram_pct:.0f}% ("
                f"{gpu_info.get('vram_used', '?')}/"
                f"{gpu_info.get('vram_total', '?')}) "
            ),
            style=style_for_percent(vram_pct, RAM_THRESHOLDS),
        )
        text.append("Temp ", style="bold")
        text.append(f"{temp_c:.1f}°C", style=style_for_percent(temp_c, TEMP_THRESHOLDS))
        return text


def show_ascii_art() -> str:
    """Return the ASCII art banner as text.

    Returns:
        The multi-line ASCII art string for the header banner.

    """
    return (
        "╦═╗  ╔═╗  ╔═╗  ╔╦╗      ╔╦╗  ╔═╗  ╔╗╔  ╦  ╔╦╗  ╔═╗  ╦═╗\n"
        "╠╦╝  ║ ║  ║    ║║║      ║║║  ║ ║  ║║║  ║   ║   ║ ║  ╠╦╝\n"
        "╩╚═  ╚═╝  ╚═╝  ╩ ╩      ╩ ╩  ╚═╝  ╝╚╝  ╩   ╩   ╚═╝  ╩╚═"
    )


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


def get_rocm_smi_output() -> dict[str, str]:
    """Execute ``rocm-smi`` and extract GPU stats for a single device.

    Returns:
        Mapping with keys: ``temp_c``, ``gpu_percent``, ``vram_percent``, and
        optionally ``vram_used``, ``vram_total``, ``name``. If unavailable,
        returns ``{"error": "..."}``.

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
            return {"error": "Could not find GPU statistics in rocm-smi output"}

        # Extract values using string splitting and indexing
        # The concise row contains multiple percentage fields; the last two
        # are typically VRAM% then GPU%.
        parts = [p for p in stats.split() if p]

        # Extract temperature (format: XX.X°C)
        temp_token = next((p for p in parts if "°C" in p), "N/A")
        percent_tokens = [p for p in parts if p.endswith("%")]
        if len(percent_tokens) >= 2:
            vram_pct_token = percent_tokens[-2]
            gpu_pct_token = percent_tokens[-1]
        else:
            # Fallbacks if unexpected format
            vram_pct_token = "0%"
            gpu_pct_token = percent_tokens[-1] if percent_tokens else "0%"

        def pct_to_float(s: str) -> float:
            try:
                return float(s.replace("%", ""))
            except Exception:
                return 0.0

        try:
            temp_c = float(temp_token.replace("°C", "").replace("C", ""))
        except Exception:
            temp_c = 0.0

        info: dict[str, str | list[dict[str, str]]]
        info = {
            "temp_c": f"{temp_c}",
            "gpu_percent": f"{pct_to_float(gpu_pct_token)}",
            "vram_percent": f"{pct_to_float(vram_pct_token)}",
            # Unknown totals from plain text; placeholders keep UI informative
            "vram_used": "N/A",
            "vram_total": "N/A",
            "name": "AMD GPU",
        }

        # Try to fetch VRAM used/total bytes from meminfo
        try:
            used_b, total_b = get_rocm_smi_vram_usage_bytes()
            if used_b is not None and total_b is not None and total_b > 0:
                info["vram_used"] = SystemMonitor.get_size(used_b)
                info["vram_total"] = SystemMonitor.get_size(total_b)
        except Exception as e:  # pragma: no cover - non-fatal enhancement
            logger.debug("Failed to parse rocm-smi meminfo: %s", e)

        # Try to gather per-process info using rocm-smi --showpids
        try:
            procs = get_rocm_smi_processes()
            if procs:
                info["processes"] = procs
        except Exception as e:  # pragma: no cover - non-fatal enhancement
            logger.debug("Failed to parse rocm-smi processes: %s", e)

        return info  # type: ignore[return-value]

    except FileNotFoundError:
        logger.error(
                "`rocm-smi` command not found. Please install ROCm SMI and "
                "ensure it's in your PATH."
        )
        return {"error": "rocm-smi command not found"}
    except subprocess.CalledProcessError as e:
        logger.error("Failed to execute rocm-smi: %s", e)
        return {"error": "Error retrieving GPU information"}
    except Exception as e:
        logger.exception("Unexpected error while executing rocm-smi: %s", e)
        return {"error": "Error retrieving GPU information"}


def get_rocm_smi_processes() -> list[dict[str, str]]:
    """Return processes using the GPU from ``rocm-smi --showpids``.

    The function is resilient to common output variants across ROCm versions.

    Returns:
        A list of process dictionaries with keys: ``pid``, ``name``, ``gpus``,
        and ``vram_used`` when available. Returns an empty list if no processes
        are detected or output is not parseable.

    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids"], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        return []

    text_out = result.stdout or ""
    lines = [ln.rstrip() for ln in text_out.splitlines() if ln.strip()]
    if not lines:
        return []

    processes: list[dict[str, str]] = []

    # Strategy A: Table format with header including PID/PROCESS
    header_idx = -1
    for i, ln in enumerate(lines):
        if (
            ("PID" in ln and "PROCESS" in ln)
            or ("PID" in ln and "NAME" in ln)
        ):
            header_idx = i
            break
    if header_idx != -1:
        # Parse rows until a non-data separator or end
        for ln in lines[header_idx + 1 :]:
            if set(ln.strip()) <= set("-="):  # separator line
                continue
            # Split on 2+ spaces to keep columns
            cols = re.split(r"\s{2,}", ln.strip())
            if len(cols) < 2:
                continue
            # Heuristic mapping based on common rocm-smi layout
            pid = cols[0]
            name = cols[1] if len(cols) > 1 else "-"
            gpus = cols[2] if len(cols) > 2 else "-"
            vram = cols[3] if len(cols) > 3 else "-"
            if pid.isdigit():
                processes.append(
                    {
                        "pid": pid,
                        "name": name,
                        "gpus": gpus,
                        "vram_used": vram,
                    }
                )
        if processes:
            return processes


def get_rocm_smi_vram_usage_bytes() -> tuple[int | None, int | None]:
    """Return VRAM used and total in bytes via ``rocm-smi --showmeminfo vram``.

    The function tries JSON output first, then falls back to text parsing
    for older ROCm outputs. It is robust to minor format differences.

    Returns:
        A tuple ``(used_bytes, total_bytes)`` where either value can be
        ``None`` if parsing fails.

    """
    # Helper to parse sizes with units like 7.5GB or 1024MB.
    def _to_bytes(token: str) -> int | None:
        try:
            m = re.match(r"(?i)^([0-9]*\.?[0-9]+)\s*([kmgtp]?b)$", token.strip())
            if not m:
                return int(token)
            val = float(m.group(1))
            unit = m.group(2).lower()
            power = {"kb": 1, "mb": 2, "gb": 3, "tb": 4, "pb": 5}.get(unit, 0)
            return int(val * (1024 ** power))
        except Exception:
            return None

    # Attempt JSON first
    try:
        res = subprocess.run(
            ["rocm-smi", "--json", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout or "{}")
        # Search recursively for meminfo fields
        used_b: int | None = None
        total_b: int | None = None

        def _walk(obj: object) -> None:
            nonlocal used_b, total_b
            if isinstance(obj, dict):
                # Common key variants
                for k, v in obj.items():
                    kl = str(k).lower()
                    if isinstance(v, (dict, list)):
                        _walk(v)
                        continue
                    if used_b is None and (
                        "used" in kl and "vram" in kl or kl.endswith("used (b)")
                    ):
                        used_b = _to_bytes(str(v))
                    if total_b is None and (
                        "total" in kl and "vram" in kl or kl.endswith("total (b)")
                    ):
                        total_b = _to_bytes(str(v))
            elif isinstance(obj, list):
                for it in obj:
                    _walk(it)

        _walk(data)
        if used_b is not None and total_b is not None:
            return used_b, total_b
    except Exception:
        # Fall through to text parsing
        pass

    # Text fallback
    try:
        res = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        return None, None
    except subprocess.CalledProcessError:
        return None, None

    used_b_txt: int | None = None
    total_b_txt: int | None = None
    for line in (res.stdout or "").splitlines():
        line_str = line.strip()
        # Try bytes first
        m_total = re.search(r"(?i)total\s+memory.*?\(b\)\s*:\s*(\d+)", line_str)
        m_used = re.search(
            r"(?i)total\s+memory\s+used.*?\(b\)\s*:\s*(\d+)", line_str
        )
        if m_total:
            try:
                total_b_txt = int(m_total.group(1))
            except Exception:
                pass
        if m_used:
            try:
                used_b_txt = int(m_used.group(1))
            except Exception:
                pass
        # Units variant (e.g., 7.5GB)
        m_total_u = re.search(
            r"(?i)total\s+memory.*?:\s*([0-9]*\.?[0-9]+\s*[kmgtp]b)", line_str
        )
        m_used_u = re.search(
            r"(?i)total\s+memory\s+used.*?:\s*([0-9]*\.?[0-9]+\s*[kmgtp]b)",
            line_str,
        )
        if m_total_u and total_b_txt is None:
            total_b_txt = _to_bytes(m_total_u.group(1))
        if m_used_u and used_b_txt is None:
            used_b_txt = _to_bytes(m_used_u.group(1))

    return used_b_txt, total_b_txt


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
@click.option("--once", is_flag=True, help="Render once and exit.")
@click.option("--no-ascii", is_flag=True, help="Disable ASCII banner.")
@click.option("--no-color", is_flag=True, help="Disable terminal colors.")
@click.option("--compact", is_flag=True, help="Compact single-line layout.")
def main(
    interval: int,
    log_file: str | None,
    once: bool,
    no_ascii: bool,
    no_color: bool,
    compact: bool,
) -> None:
    """Run the system monitoring application.

    Args:
        interval: Update interval in seconds
        log_file: Optional path to log file
        once: Render a single update and exit.
        no_ascii: Disable the startup ASCII banner.
        no_color: Disable terminal colors.
        compact: Use compact one-line/non-layout rendering.

    Raises:
        click.BadParameter: If the provided interval is not a positive integer.

    """
    try:
        if interval <= 0:
            raise click.BadParameter("Interval must be a positive integer.")

        setup_logging(log_file)
        check_rocm_smi_installed()
        check_system_compatibility()

        console = Console(theme=THEME, no_color=no_color)
        if not console.is_terminal:
            compact = True
            if not once:
                once = True

        monitor = SystemMonitor(
            interval, compact=compact, no_ascii=no_ascii, console=console
        )
        setattr(monitor, "_once", once)
        monitor.run()

    except click.BadParameter as e:
        logger.error("Parameter error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
