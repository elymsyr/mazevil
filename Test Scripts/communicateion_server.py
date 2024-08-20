import socket

# Set up the socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)
sock.bind(serverAddressPort)

print(f"Listening for UDP data on {serverAddressPort}...")

try:
    while True:
        # Receive data from the socket
        data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
        print(f"Received message: {data.decode('utf-8')} from {addr}")
except KeyboardInterrupt:
    print("\nServer stopped.")
finally:
    sock.close()
