import yaml
import argparse
import numpy as np
import math
from fractions import Fraction
from collections import namedtuple
import json
import gzip
import logging

from lifcon import (
    Move, Wait, ControllerStatus, Person, World,
    STATE_GOING_UP, STATE_GOING_DOWN, STATE_EMPTY)

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()

#世界設定ファイルのパス→環境？
parser.add_argument('--world', type=str,
                    help='world configuration')
#リプレイファイルのパス
parser.add_argument('--replay', type=str, default="/dev/null",
                    help='replay file (gziped json)')
#DQNパラメータのロード
parser.add_argument('--dqnparam', type=str, default=None,
                    help='Read DQN parameter, and use DQN')
#ε-greedy法に用いるεの定義
parser.add_argument('--epsilon', type=float, default=0.0,
                    help='Epsilon value used for DQN eps-greedy')
#start tick???
parser.add_argument('--starttick', type=int, default=0,
                    help='Start tick')
#リモっとtick
parser.add_argument('--limittick', type=int, default=None,
                    help='Limit the number of ticks for debugging purposes')
#使用シード
parser.add_argument('--seed', type=int, default=0,
                    help='seed')
#ランダムticks
parser.add_argument('--random-ticks', type=int, default=None,
                    help='Limit the number of ticks for debugging purposes')


opt = parser.parse_args()
np.random.seed(opt.seed)

#アクションオブジェクトをJSON形式に変換
def action_to_jsval(a):
    if isinstance(a, Move):
        return float(a.dest)
    elif isinstance(a, Wait):
        return "STOP"
    else:
        raise ValueError("Unknown action type")

#人物オブジェクトをJSON形式に変換
def person_to_jsval(p):
    return {
        "sf": float(p.spawn_floor),
        "st": float(p.spawn_tick),
        "df": float(p.dest_floor)
    }

#JSONエンコード時にシリアライズするカスタムエンコーダー
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        elif isinstance(obj, Fraction):
            return [obj.numerator, obj.denominator]

        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

#エレベータのコントローラーの定義
class RuleBasedController:
    def __init__(self, world):
        self.world = world
        self.nlifts = self.world.nlifts

    def tick(self, status, tick):
        goals = []
        for lid in range(self.nlifts):
            loc = status.locations[lid]
            if len(status.members[lid]) == 0:
                # If the lift is empty,
                # move to the highest/ lowest floor with waiting people

                highf = None
                lowf = None
                for fid in range(len(status.wait_up)):
                    if status.wait_up[fid] or status.wait_down[fid]:
                        if highf is None or fid > highf:
                            highf = fid
                        if lowf is None or fid < lowf:
                            lowf = fid
                dest = lowf if lid % 2 == 0 else highf
                if dest == None:
                    goals.append(Wait())
                else:
                    goals.append(Move(dest))
            else:
                _, dest = min((abs(loc - p.dest_floor), p.dest_floor)
                              for p in status.members[lid])
                # find stop-by
                if dest > loc: # Going-up
                    ceil_loc = math.ceil(loc)
                    for fid, flag in list(enumerate(status.wait_up))[ceil_loc:dest]:
                        if flag:
                            dest = fid
                else: # Going-down
                    floor_loc = math.floor(loc)
                    for fid, flag in list(enumerate(status.wait_down))[dest + 1:floor_loc:-1]:
                        if flag:
                            dest = fid
                goals.append(Move(dest))
        return goals, [True for _ in range(self.nlifts)]

#シミュレーターの定義
class Simulator:
    def __init__(self, world, controller, replayout):
        self.world = world
        self.controller = controller
        self.nlifts = self.world.nlifts
        self.nfloors = self.world.nfloors
        self.lift_speed = Fraction(1, self.world.lift_inv_speed)

        self.max_people_per_lift = self.world.max_people_per_lift

        self.replayout = replayout

        self.reset() # Initialize state variables

    def reset(self):
        self.time = 0
        self.locations = [Fraction(0) for _ in range(self.nlifts)]

        self.waiting_up = [[] for _ in range(self.nfloors)]
        self.waiting_down = [[] for _ in range(self.nfloors)]

        self.stop_time = [0 for _ in range(self.nlifts)]
        # A lift has two kinds of states, one is accept_state that decides
        # whether the lift will make a stop requested by an external customer.
        # Another kind is move_state that denotes the moving directions.
        # The former is controlled by the controller (DQN), whereas the later is
        # basically controlled by the people in the lift.
        self.accept_state = [True for _ in range(self.nlifts)]
        self.move_state = [STATE_EMPTY for _ in range(self.nlifts)]

        self.people_in_lift = [[] for _ in range(self.nlifts)]

    def reset_random_locations(self):
        self.locations = [Fraction(np.random.randint(0, self.nfloors))
                          for _ in range(self.nlifts)]

    def _write_replay_json(self, s):
        if isinstance(s, str):
            self.replayout.write(s.encode('utf-8'))
        else:
            self.replayout.write(s)

    def _decide_move_direction(self, wait_up, wait_down):
        """
        Decide direction, when the lift is empty and there's at least one person
        """
        if len(wait_up) == 0 and len(wait_down) == 0:
            assert False, "This shouldn't happen"
        elif len(wait_up) == 0:
            return STATE_GOING_DOWN
        elif len(wait_down) == 0:
            return STATE_GOING_UP
        else:
            if wait_up[0].spawn_tick < wait_down[0].spawn_tick:
                return STATE_GOING_UP
            else:
                return STATE_GOING_DOWN


    def ticks(self, starttick, nticks):
        current_reward = 0.0
        num_spawned = 0

        self._write_replay_json("[")

        lasttick = min(self.world.ticks_per_day, starttick + nticks)
        logging.info("Start-tick=%07d, last-tick=%07d (nticks=%07d)",
                     starttick, lasttick, nticks)

        for tick in range(starttick, lasttick):
            if tick != starttick:
                self._write_replay_json(",")

            tickinfo = {'tick': tick}

            # Spawn people
            incoming = world.spawn_people(tick)

            for fid, l in enumerate(incoming):
                if len(l) == 0:
                    continue
                num_spawned += len(l)

                self.waiting_up[fid].extend(filter(lambda p: p.dest_floor > fid, l))
                self.waiting_down[fid].extend(filter(lambda p: p.dest_floor < fid, l))

            tickinfo['acc_npeople'] = num_spawned

            # Let people out
            #  In-lift people can get off regardless of the lift accept policy,
            #  Furthermore, when a lift is stopping, people can jump in regardless
            #  of accept policy.
            rewards = []
            for lid in range(self.nlifts):
                if self.locations[lid].denominator != 1:
                    continue

                loc = self.locations[lid].numerator
                exit_people = list(filter(
                    lambda p: p.dest_floor == loc,
                    self.people_in_lift[lid]))

                if len(exit_people) == 0:
                    continue

                # Make a stop
                self.stop_time[lid] = self.world.lift_stop_duration
                self.people_in_lift[lid] = list(filter(
                    lambda p: p.dest_floor != loc,
                    self.people_in_lift[lid]))

                # Compute transition speeds and rewards
                #報酬の計算
                for p in exit_people:
                    dur = tick - p.spawn_tick
                    dfloor = abs(p.dest_floor - p.spawn_floor)

                    eff_speed = Fraction(dfloor, dur)
                    reward = eff_speed / self.lift_speed

                    rewards.append(float(reward))
                    current_reward += reward

                # If the lift becomes empty, reset move-state
                if len(self.people_in_lift[lid]) == 0:
                    self.move_state[lid] = STATE_EMPTY

            tickinfo['rewards'] = rewards
            tickinfo['acc_rewards'] = current_reward
            logging.info("Ticks=%07d, total reward=%f, avg. reward=%f",
                         tick, current_reward, current_reward / (num_spawned + 0.001))

            # Let people in
            for lid in range(self.nlifts):
                if self.locations[lid].denominator != 1:
                    continue

                loc = self.locations[lid].numerator

                if (len(self.waiting_up[loc]) == 0 and
                    len(self.waiting_down[loc]) == 0):
                    continue # Nobody's waiting

                movdir = self.move_state[lid]
                if movdir == STATE_EMPTY:
                    movdir = self._decide_move_direction(self.waiting_up[loc],
                                                         self.waiting_down[loc])

                vacancy = self.max_people_per_lift - len(self.people_in_lift[lid])

                is_waiting = (len(self.waiting_up[loc]) > 0
                              if movdir == STATE_GOING_UP else
                              len(self.waiting_down[loc]) > 0)

                if is_waiting and self.accept_state[lid]:
                    # Make a stop
                    if vacancy > 0: # if empty
                        self.stop_time[lid] = self.world.lift_stop_duration
                    #elif self.stop_time[lid] == 0:
                    #    # Even if the lift is full, at least it opens the door
                    #    # for reality
                    #    self.stop_time[lid] = self.world.lift_stop_duration
                if self.stop_time[lid] >= 0 and is_waiting:
                    if movdir == STATE_GOING_UP: # Move upward people in
                        mov = min(vacancy, len(self.waiting_up[loc]))
                        self.people_in_lift[lid] += self.waiting_up[loc][:mov]
                        self.waiting_up[loc] = self.waiting_up[loc][mov:]
                    else: # Move downward people in
                        mov = min(vacancy, len(self.waiting_down[loc]))
                        self.people_in_lift[lid] += self.waiting_down[loc][:mov]
                        self.waiting_down[loc] = self.waiting_down[loc][mov:]

                    # When accepting a person, fix the moving direction
                    if mov > 0:
                        self.move_state[lid] = movdir


            tickinfo['wait_dn'] = [len(l) for l in self.waiting_down]
            tickinfo['wait_up'] = [len(l) for l in self.waiting_up]
            tickinfo['stop_time'] = self.stop_time
            tickinfo['statuses'] = self.move_state

            tickinfo['locations'] = self.locations
            tickinfo['inlift_p'] = [[person_to_jsval(p) for p in l]
                                    for l in self.people_in_lift]

            # Make action
            goals, acceptst = self.controller.tick(
                ControllerStatus([len(l) != 0 for l in self.waiting_up],
                                 [len(l) != 0 for l in self.waiting_down],
                                 self.locations,
                                 self.people_in_lift,
                                 self.move_state),
                tick)
            tickinfo['goals'] = [action_to_jsval(a) for a in goals]
            tickinfo['acceptst'] = acceptst

            self.accept_state = acceptst

            # Move lifts
            for lid in range(self.nlifts):
                if self.stop_time[lid]:
                    self.stop_time[lid] -= 1
                    continue

                if self.move_state[lid] == STATE_GOING_UP:
                    self.locations[lid] += self.lift_speed
                    assert self.locations[lid] <= self.nfloors
                elif self.move_state[lid] == STATE_GOING_DOWN:
                    self.locations[lid] -= self.lift_speed
                    assert self.locations[lid] >= 0
                else:
                    if isinstance(goals[lid], Wait):
                        continue
                    elif isinstance(goals[lid], Move):
                        if self.locations[lid] > goals[lid].dest:
                            self.locations[lid] -= self.lift_speed
                        elif self.locations[lid] < goals[lid].dest:
                            self.locations[lid] += self.lift_speed

            self._write_replay_json(json.dumps(tickinfo, cls=CustomEncoder,
                                               separators=(',', ':')))

            yield tick

        self._write_replay_json("]")

print(opt)

world_conf = yaml.load(open(opt.world).read())

world = World(world_conf)
if not opt.dqnparam:
    controller = RuleBasedController(world)
else:
    from lifcon_dqn import DQNController
    controller = DQNController(world, opt.dqnparam, opt.epsilon)

with gzip.open(opt.replay, "w") as replayout:
    simulator = Simulator(world, controller, replayout)

    nticks = world.ticks_per_day
    start_tick = opt.starttick

    if opt.random_ticks is not None:
        start_tick = np.random.randint(0, world.ticks_per_day - opt.random_ticks)
        nticks = opt.random_ticks
        simulator.reset_random_locations()

    if opt.limittick is not None:
        nticks = min(nticks, opt.limittick)

    for tick in simulator.ticks(start_tick, nticks):
        pass