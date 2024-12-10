# ROCm Monitor

A Python-based system monitoring tool that provides real-time GPU, CPU, and RAM statistics through a beautiful terminal interface, specifically designed for AMD GPUs running ROCm.

## Versions

**Current version**: 1.0.0

## Table of Contents

- [Versions](#versions)
- [Badges](#badges)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## Badges

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/platform-linux-lightgrey.svg)

## Installation

1. Create and activate a virtual environment:

    ```bash
    python3.11 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:

    ```bash
    pip install rich psutil click
    ```

3. Install ROCm and `rocm-smi` following the official guides:

    - [Arch Linux](https://wiki.archlinux.org/title/AMDGPU#ROCm)
    - [Ubuntu](https://rocmdocs.amd.com/en/latest/Installation_Guide/Ubuntu.html)

4. Make the script executable:

    ```bash
    chmod +x rocm-mon.py
    ```

## Usage

Run the script from the virtual environment:

```bash
./rocm-mon.py
```

### Command-Line Options

- `-i, --interval`: Update interval in seconds (default: 5)
- `-l, --log-file`: File path to save logs
- `--help`: Show help message

### Examples

```bash
# Update every 2 seconds
./rocm-mon.py -i 2

# Save logs to file
./rocm-mon.py -l monitor.log
```

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for more information.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
