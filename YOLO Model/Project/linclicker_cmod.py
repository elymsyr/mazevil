def run_commands():
    import subprocess
    try:
        # Change permissions
        subprocess.run(['sudo', 'chmod', '666', '/dev/uinput'], 
                       check=True, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)
        print("Permissions for /dev/uinput changed successfully.")

        # Load uinput module
        subprocess.run(['sudo', 'modprobe', 'uinput'], 
                       check=True, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE,
                       text=True)
        print("uinput module loaded successfully.")

    except subprocess.CalledProcessError as e:
        print("Error executing command:", e.stderr)