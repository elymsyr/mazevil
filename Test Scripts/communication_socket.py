import socket

# Set up the socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)

try:
    # Send a message to the server
    message = "Hello, UDP server!111"
    sock.sendto(message.encode('utf-8'), serverAddressPort)
    print(f"Message sent to {serverAddressPort}")
except Exception as e:
    print(f"Error: {e}")
finally:
    sock.close()
