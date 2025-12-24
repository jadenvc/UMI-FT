import subprocess


def kill_processes_on_ports(ports):
    """
    Kills any processes using the specified TCP port numbers.

    Args:
        ports (list[int] or list[str]): List of port numbers.
    """
    for port in ports:
        try:
            # Find the PID using the porta
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            pids = result.stdout.strip().split("\n")
            pids = [pid for pid in pids if pid]

            for pid in pids:
                print(f"Killing process {pid} on port {port}...")
                subprocess.run(["kill", "-9", pid], check=False)
        except Exception as e:
            print(f"Failed to kill process on port {port}: {e}")


if __name__ == "__main__":
    # Example usage
    ports_to_kill = [1000, 5555, 5556]  # Add your ports here
    kill_processes_on_ports(ports_to_kill)
