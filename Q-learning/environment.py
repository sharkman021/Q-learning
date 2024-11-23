# environment.py

import numpy as np
import random
from collections import deque

class Passenger:
    def __init__(self, spawn_floor, dest_floor, spawn_time):
        self.spawn_floor = spawn_floor
        self.dest_floor = dest_floor
        self.spawn_time = spawn_time
        self.pickup_time = None
        self.dropoff_time = None

class Elevator:
    def __init__(self, eid, current_floor=1, direction='idle'):
        self.eid = eid
        self.current_floor = current_floor
        self.direction = direction  # 'up', 'down', 'idle'
        self.passengers = []
        self.capacity = 10
        self.next_stops = set()

    def move(self):
        if self.direction == 'up':
            self.current_floor += 1
        elif self.direction == 'down':
            self.current_floor -= 1

    def update_direction(self):
        if len(self.passengers) == 0 and not self.next_stops:
            self.direction = 'idle'
        else:
            if self.direction == 'up':
                if any(floor > self.current_floor for floor in self.next_stops):
                    self.direction = 'up'
                elif any(floor < self.current_floor for floor in self.next_stops):
                    self.direction = 'down'
                else:
                    self.direction = 'idle'
            elif self.direction == 'down':
                if any(floor < self.current_floor for floor in self.next_stops):
                    self.direction = 'down'
                elif any(floor > self.current_floor for floor in self.next_stops):
                    self.direction = 'up'
                else:
                    self.direction = 'idle'
            elif self.direction == 'idle':
                if self.next_stops:
                    if min(self.next_stops) > self.current_floor:
                        self.direction = 'up'
                    elif max(self.next_stops) < self.current_floor:
                        self.direction = 'down'
                    else:
                        # Choose direction based on nearest stop
                        nearest_up = min([floor for floor in self.next_stops if floor > self.current_floor], default=None)
                        nearest_down = max([floor for floor in self.next_stops if floor < self.current_floor], default=None)
                        if nearest_up and nearest_down:
                            if abs(nearest_up - self.current_floor) < abs(self.current_floor - nearest_down):
                                self.direction = 'up'
                            else:
                                self.direction = 'down'
                        elif nearest_up:
                            self.direction = 'up'
                        elif nearest_down:
                            self.direction = 'down'

class ElevatorEnv:
    def __init__(self, config):
        self.num_floors = config['num_floors']
        self.num_lifts = config['num_lifts']
        self.tick_per_day = config['tick_per_day']
        self.max_people_per_lift = config['max_people_per_lift']
        self.lift_move_time = config['lift_move_time']
        self.lift_stop_duration = config['lift_stop_duration']
        
        self.spawn_lambdas = config['spawn_lambdas']
        self.destination_probs = config['destination_probs']
        
        self.elevators = [Elevator(eid=i+1) for i in range(self.num_lifts)]
        
        # Waiting passengers: {floor: deque}
        self.waiting_up = {floor: deque() for floor in range(1, self.num_floors+1)}
        self.waiting_down = {floor: deque() for floor in range(1, self.num_floors+1)}
        
        self.current_time = 0
        self.passengers = []
        
    def reset(self):
        self.elevators = [Elevator(eid=i+1) for i in range(self.num_lifts)]
        self.waiting_up = {floor: deque() for floor in range(1, self.num_floors+1)}
        self.waiting_down = {floor: deque() for floor in range(1, self.num_floors+1)}
        self.current_time = 0
        self.passengers = []
        
    def step(self, actions):
        """
        actions: list of actions for each elevator
                 each action is 'up', 'down', or 'idle'
        """
        rewards = 0
        done = False
        
        # Move elevators based on actions
        for elevator, action in zip(self.elevators, actions):
            if action == 'up' and elevator.current_floor < self.num_floors:
                elevator.direction = 'up'
                elevator.move()
            elif action == 'down' and elevator.current_floor > 1:
                elevator.direction = 'down'
                elevator.move()
            else:
                elevator.direction = 'idle'
        
        # Let passengers out
        for elevator in self.elevators:
            departing_passengers = [p for p in elevator.passengers if p.dest_floor == elevator.current_floor]
            for p in departing_passengers:
                p.dropoff_time = self.current_time
                rewards += 1  # Reward for successful drop-off
                elevator.passengers.remove(p)
            # Update direction if no passengers
            elevator.update_direction()
        
        # Let passengers in
        for elevator in self.elevators:
            if elevator.direction == 'up' or elevator.direction == 'idle':
                queue = self.waiting_up[elevator.current_floor]
            elif elevator.direction == 'down':
                queue = self.waiting_down[elevator.current_floor]
            else:
                queue = deque()
            
            while len(elevator.passengers) < self.max_people_per_lift and queue:
                passenger = queue.popleft()
                passenger.pickup_time = self.current_time
                elevator.passengers.append(passenger)
                elevator.next_stops.add(passenger.dest_floor)
                rewards -= 1  # Penalty for waiting time
        
        # Generate new passengers
        for floor in range(1, self.num_floors+1):
            lambda_val = self.spawn_lambdas.get(str(floor), 1)
            num_new = np.random.poisson(lambda_val)
            for _ in range(num_new):
                possible_floors = [f for f in range(1, self.num_floors+1) if f != floor]
                dest_probs_full = self.destination_probs.get(str(floor), [0]*self.num_floors)
                dest_probs = [dest_probs_full[f-1] for f in possible_floors]
                total_prob = sum(dest_probs)
                if total_prob == 0:
                    # All probabilities zero, assign uniform probabilities
                    dest_probs = [1.0 / len(possible_floors)] * len(possible_floors)
                else:
                    # Normalize probabilities
                    dest_probs = [p / total_prob for p in dest_probs]
                dest_floor = np.random.choice(possible_floors, p=dest_probs)
                passenger = Passenger(spawn_floor=floor, dest_floor=dest_floor, spawn_time=self.current_time)
                self.passengers.append(passenger)
                if dest_floor > floor:
                    self.waiting_up[floor].append(passenger)
                else:
                    self.waiting_down[floor].append(passenger)
        
        self.current_time += 1
        
        # Check if simulation time is over
        if self.current_time >= self.tick_per_day:
            done = True
        
        # Calculate total waiting time
        total_waiting = sum(len(queue) for queue in self.waiting_up.values()) + sum(len(queue) for queue in self.waiting_down.values())
        
        return self._get_state(), rewards, done, total_waiting
    
    def _get_state(self):
        state = []
        for elevator in self.elevators:
            state.append(elevator.current_floor)
            state.append(elevator.direction)
        for floor in range(1, self.num_floors+1):
            state.append(len(self.waiting_up[floor]))
            state.append(len(self.waiting_down[floor]))
        return tuple(state)
    
    def render(self):
        print(f"Time: {self.current_time}")
        for elevator in self.elevators:
            print(f"Elevator {elevator.eid}: Floor {elevator.current_floor}, Direction {elevator.direction}, Passengers {len(elevator.passengers)}")
        for floor in range(1, self.num_floors+1):
            print(f"Floor {floor}: Up {len(self.waiting_up[floor])}, Down {len(self.waiting_down[floor])}")
        print("-" * 50)
