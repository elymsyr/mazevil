import multiprocessing
import time

class Game:
    def __init__(self):
        self.move = "Player moving"

    def move_player(self):
        while True:
            print(self.move)
            time.sleep(1)  # Adjust the sleep time as needed

    def start_movement(self):
        # Create a process to run the move_player method
        # Ensure `move_player` does not depend on unpickleable attributes
        process = multiprocessing.Process(target=self.move_player)
        process.start()
        return process

    def stop_movement(self, process):
        # Terminate the process
        process.terminate()
        process.join()

if __name__ == '__main__':
    game = Game()
    movement_process = game.start_movement()
    
    # Simulate some other work in the main process
    time.sleep(10)  # Main process doing something else for 10 seconds
    
    # Stop the movement process
    game.stop_movement(movement_process)
