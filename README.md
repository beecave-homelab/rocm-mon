# ROCm Monitor

This Python script continuously monitors GPU statistics using the `rocm-smi` command along with CPU and RAM usage, presenting the data in a visually appealing format using the `rich` library. It’s particularly useful for tracking system performance metrics on systems equipped with AMD GPUs running ROCm.

## Repository

GitHub Repository: [rocm-mon](https://github.com/beecave-homelab/rocm-mon.git)

## Description

ROCm Monitor collects real-time data on your GPU, CPU, and memory usage and displays it in a dynamic, easy-to-read dashboard. The tool leverages `rocm-smi` for GPU statistics and `psutil` for CPU and RAM data, making it suitable for systems using AMD’s ROCm platform.

Each monitoring session displays GPU utilization, current memory usage, and CPU load at a user-defined refresh interval. The script also features error handling to manage scenarios like missing `rocm-smi` installation, unexpected process failures, or system-level errors gracefully.

## Table of Contents

- [Versions](#versions)
- [Badges](#badges)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## Versions

**Current Version**: 1.0.0  
Supports Python 3.8+, `rocm-smi` for AMD GPU monitoring, and uses `rich` for visual output.

## Badges

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/beecave-homelab/rocm-mon.git
   cd rocm-mon
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ and install required packages:
   ```bash
   pip install rich psutil
   ```

3. **Install ROCm and `rocm-smi`**:
   - Make sure that `rocm-smi` is installed and accessible in your system’s PATH. 
   - You can follow the official ROCm installation guide [here](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html).

## Usage

To run the script with default settings:

```bash
python rocm-mon.py
```

This displays the real-time GPU, CPU, and RAM usage with a 5-second refresh interval. ASCII art is shown at the beginning of the session for a visual banner.

### Command-Line Options

- **`-i, --interval`**: Set the refresh interval in seconds (default: 5).
- **`-l, --log-file`**: Specify a log file to save logs. Example: `-l monitor.log`.
- **`-h, --help`**: Show help message with ASCII art and usage information.

### Example Commands

1. **Run with a 10-second interval**:
   ```bash
   python rocm-mon.py -i 10
   ```

2. **Log output to `monitor.log`**:
   ```bash
   python rocm-mon.py -l monitor.log
   ```

3. **Help Message**:
   ```bash
   python rocm-mon.py -h
   ```

This displays ASCII art, help message, and example usage instructions.

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for more information.

## Contributing

Contributions are welcome! If you’d like to improve the script or suggest features, please open an issue or create a pull request in the repository. For major changes, please discuss them in an issue first.

---

Enjoy real-time monitoring with ROCm Monitor, and gain better insights into your system’s performance!