# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import copy
import cv2
from typing import Dict

from gymnasium import Wrapper, Env

ORIENTATION_DIF_TOLERANCE = 0.2
AREA_DIF_TOLERANCE = 16

SPEED_DIF_TOLERANCE = 6
MAX_SPEED = 50


class Vector2():
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


    def area(self):
        return abs(self.x * self.y)


    def manhat_length(self):
        return abs(self.x) + abs(self.y)


    def __sub__(self, other):
        return Vector2(self.x - other.x, self.y - other.y)
    

    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)


    def __mul__(self, other):
        return Vector2(self.x * other, self.y * other)


    def __truediv__(self, other):
        return Vector2(self.x / other, self.y / other)

    
    def __eq__(self, value: object) -> bool:
        return self.x == value.x and self.y == value.y


    def __str__(self) -> str:
        return f"({self.x},{self.y})"

    def __repr__(self) -> str:
        return f"({self.x},{self.y})"


class Object():
    def __init__(self, position: Vector2, size: Vector2, orientation: float, color: int):
        self.position = position
        self.size = size
        self.orientation = orientation
        self.color = color
        self.category = None


    def __repr__(self) -> str:
        return f"obj @ {self.position} -> [Area: {self.size.area()}, Solidity: {self.orientation}]"


class ObjectCategory():
    def __init__(self, size: Vector2, orientation: float, indicator_color: np.array) -> None:
        self.size = size
        self.orientation = orientation
        self.indicator_color = indicator_color
    
    
    def belongs(self, object: Object) -> bool:
        return abs(self.size.area() - object.size.area()) <= AREA_DIF_TOLERANCE and abs(self.orientation - object.orientation) <= ORIENTATION_DIF_TOLERANCE
    

    @staticmethod
    def from_obj(object: Object):
        return ObjectCategory(object.size, object.orientation, np.random.randint(256, size=(3)))


    def __repr__(self) -> str:
        return f"{self.size}"


class Event():
    def __init__(self, obj_category: str, category: str, current_pos: Vector2) -> None:
        self.obj_category = obj_category
        self.category = category
        self.current_pos = current_pos


class AppearanceEvent(Event):
    def __init__(self, obj_category: str, current_pos: Vector2) -> None:
        super().__init__(obj_category, "APPEARANCE", current_pos)
    

    def __repr__(self) -> str:
        return f"app @ {self.current_pos}"


class DisappearanceEvent(Event):
    def __init__(self, obj_category: str, current_pos: Vector2) -> None:
        super().__init__(obj_category, "DISAPPEARANCE", current_pos)
    

    def __repr__(self) -> str:
        return f"dis @ {self.current_pos}"


class MovementEvent(Event):
    def __init__(self, obj_category: str, initial_pos: Vector2, initial_timestep: int, current_pos: Vector2):
        super().__init__(obj_category, "MOVEMENT", current_pos)
        self.initial_pos = initial_pos
        self.initial_timestep = initial_timestep


    def get_vel(self, current_timestep: int) -> Vector2:
        return (self.current_pos - self.initial_pos) / float(current_timestep - self.initial_timestep)


    def get_state(self, current_timestep: int):
        """
        Given a timestep, returns the current position of this event and its average velocity
        """
        return self.current_pos, self.get_vel(current_timestep)


    def __repr__(self) -> str:
        return f"{self.initial_pos} -> {self.current_pos}"


class EpisodeTracker():
    def __init__(self, background_colors: np.array, relevant_cat_count: int=2, headless: bool=False) -> None:
        self.background_colors = background_colors

        self.headless = headless
        self.object_categories: list[ObjectCategory] = []
        self.tracked_objects: list[Object] = []
        self.tracked_events: list[Event] = []
        self.timestep: int = 0
        
        self.category_counts: dict[ObjectCategory, int] = {}
        self.total_category_appearances = 0
        self.relevant_cat_count = relevant_cat_count


    def process_frame(self, data: np.array) -> list[Event]:
        separated_bg = self.background_separation(data)
        objs = self.object_identification(separated_bg)
        objs = self.object_categorization(objs)
        transitions = self.object_tracking(objs)
        events = self.event_tracking(objs, transitions)
        filtered_events, valid_categories = self.filtering(events)

        self.tracked_objects = objs
        self.tracked_events = events

        self.timestep += 1

        # Render mode

        if not self.headless:
            for obj in objs:
                top_left = obj.position
                bottom_right = obj.position + obj.size
                cv2.rectangle(separated_bg, (top_left.x, top_left.y), (bottom_right.x, bottom_right.y), tuple(map(int, obj.category.indicator_color)), 2)
            for obj in transitions:
                cv2.arrowedLine(separated_bg, (obj.position.x, obj.position.y), (transitions[obj].position.x, transitions[obj].position.y), (255, 255, 255), 2)
            for event in filtered_events:
                if event.category == "APPEARANCE":
                    cv2.circle(separated_bg, (event.current_pos.x, event.current_pos.y), 4, (0, 255, 0), 2)
                elif event.category == "DISAPPEARANCE":
                    cv2.circle(separated_bg, (event.current_pos.x, event.current_pos.y), 4, (255, 0, 0), 2)
                else:
                    cv2.arrowedLine(separated_bg, (event.initial_pos.x, event.initial_pos.y), (event.current_pos.x, event.current_pos.y), (0, 0, 255), 2)
        return separated_bg, filtered_events, valid_categories


    def get_event_vel(self, event: MovementEvent) -> Vector2:
        return event.get_vel(self.timestep)


    def background_separation(self, data: np.array, threshold=1) -> np.array:
        mask = np.ones(data.shape[:2], dtype=np.uint8) * 255

        for color in self.background_colors:
            lower_bound = np.array(color) - threshold
            upper_bound = np.array(color) + threshold

            lower_bound = np.clip(lower_bound, 0, 255)
            upper_bound = np.clip(upper_bound, 0, 255)

            color_mask = cv2.inRange(data, lower_bound, upper_bound)

            mask[color_mask > 0] = 0
        
        result = cv2.bitwise_and(data, data, mask=mask)

        return result

    
    def object_identification(self, data: np.array) -> list[Object]:
        r, _, _ = cv2.split(data)
        peaks = np.unique(r)

        objects = []
        for i, peak in enumerate(peaks):
            if peak == 0: # We're not considering black
                continue 
            
            peak = np.array(peak)
            mask = cv2.inRange(r, peak, peak)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for j, contour in enumerate(contours):
                bbox = cv2.boundingRect(contour)
                if bbox[2] <= 1 or bbox[3] <= 1 or bbox[0] == 0 or bbox[1] == 0 or bbox[0] + bbox[2] >= r.shape[1] or bbox[1] + bbox[3] >= r.shape[0]:
                    continue

                orientation = cv2.contourArea(contour) / (bbox[2] * bbox[3])
                objects.append(Object(Vector2(bbox[0], bbox[1]), Vector2(bbox[2], bbox[3]), orientation, r))

        return objects


    def update_category_importance(self, category: ObjectCategory):
        self.category_counts[category] += 1
        self.total_category_appearances += 1


    def get_category_importance(self, category: ObjectCategory) -> float:
        return self.category_counts[category] / self.total_category_appearances

    
    def object_categorization(self, data: list[Object]) -> list[Object]:
        for obj in data:
            for cat in self.object_categories:
                if cat.belongs(obj):
                    obj.category = cat
                    break
            if obj.category is None:
                new_cat = ObjectCategory.from_obj(obj)
                self.object_categories.append(new_cat)
                obj.category = new_cat

                self.category_counts[new_cat] = 0
            self.update_category_importance(obj.category)

        return data


    def get_closest_object(self, pos: Vector2, l: list[Object]) -> Object:
        best_distance = (l[0].position - pos).manhat_length()
        best_obj = l[0]
        
        for i in range(1, len(l)):
            new_distance = (l[i].position - pos).manhat_length()
            if new_distance < best_distance:
                best_distance = new_distance
                best_obj = l[i]
        
        return best_obj


    def put_closest_object(self, expected_positions, object_transitions_base, overfilled_objects, obj, dest_list):
        if len(dest_list) == 0:
            return

        closest_object = self.get_closest_object(expected_positions[obj], dest_list)
        if closest_object in object_transitions_base:
            if not closest_object in overfilled_objects:
                overfilled_objects.append(closest_object)

            object_transitions_base[closest_object].append(obj)
        else:
            object_transitions_base[closest_object] = [obj]


    def object_tracking(self, data: list[Object]) -> Dict[Object, Object]:
        expected_positions = {}
        for obj in self.tracked_objects:
            expected_positions[obj] = obj.position
            if obj in self.tracked_events and self.tracked_events[obj].category == "MOVEMENT":
                expected_positions[obj] += self.tracked_events[obj].get_vel(self.timestep)

        new_objs = copy.copy(data)
        object_transitions_base = {} # Dest obj to source objs that considered it closest obj
        overfilled_objects = []
        for obj in self.tracked_objects:
            self.put_closest_object(expected_positions, object_transitions_base, overfilled_objects, obj, new_objs)

        while len(overfilled_objects) > 0:
            obj = overfilled_objects.pop()
            closest_dist = (obj.position - object_transitions_base[obj][0].position).manhat_length()
            closest_obj = object_transitions_base[obj][0]
            for i in range(1, len(object_transitions_base[obj])):
                new_dist = (obj.position - object_transitions_base[obj][i].position).manhat_length()
                if new_dist < closest_dist:
                    closest_dist = new_dist
                    closest_obj = object_transitions_base[obj][i]
            
            new_objs.remove(obj)
            for i in range(0, len(object_transitions_base[obj])):
                if object_transitions_base[obj][i] != closest_obj:
                    self.put_closest_object(expected_positions, object_transitions_base, overfilled_objects, object_transitions_base[obj][i], new_objs)

            object_transitions_base[obj] = [closest_obj]
        
        object_transitions = {}
        for obj in object_transitions_base:
            object_transitions[object_transitions_base[obj][0]] = obj
        
        return object_transitions


    def event_tracking(self, objects: list[Object], transitions: Dict[Object, Object]) -> list[Event]:
        events = []

        dests = transitions.values()
        for obj in objects:
            if not obj in dests:
                events.append(AppearanceEvent(obj.category, obj.position))
        sources = transitions.keys()
        for obj in self.tracked_objects:
            if not obj in sources:
                events.append(DisappearanceEvent(obj.category, obj.position))
        
        events_by_pos: Dict[str, Event] = {}
        for event in self.tracked_events:
            events_by_pos[str(event.current_pos)] = event
        for obj in transitions:
            event = events_by_pos[str(obj.position)]
            if event.category == "APPEARANCE":
                events.append(MovementEvent(obj.category, event.current_pos, self.timestep - 1, transitions[obj].position))
            else:
                vel = transitions[obj].position - obj.position
                if (vel - event.get_vel(self.timestep)).manhat_length() < SPEED_DIF_TOLERANCE and vel.manhat_length() != 0:
                    event.current_pos = transitions[obj].position
                    events.append(event)
                else:
                    events.append(MovementEvent(obj.category, obj.position, self.timestep - 1, transitions[obj].position))

        return events


    def filtering(self, events: list[Event]) -> list[Event]:
        # Order categories
        categories = list(self.category_counts.keys())

        for i in range(1, len(categories)):
            key = categories[i]
            j = i - 1

            while j >= 0 and self.get_category_importance(key) > self.get_category_importance(categories[j]):
                categories[j + 1] = categories[j]
                j -= 1
            categories[j + 1] = key

        # Delete all non relevant categories and overly speedy movement events
        filtered_events = events.copy()
        valid_categories = categories[:self.relevant_cat_count]
        pos = 0
        while pos < len(filtered_events):
            if not filtered_events[pos].obj_category in valid_categories or (filtered_events[pos].category == "MOVEMENT" and filtered_events[pos].get_vel(self.timestep).manhat_length() > MAX_SPEED): 
                filtered_events.pop(pos)
            else:
                pos += 1

        return filtered_events, valid_categories

    
    def finish_episode(self) -> None:
        self.tracked_objects = []
        self.tracked_events = []
        self.timestep = 0

class AtariWrapper(Wrapper):
    def __init__(self, env: Env, frames_per_action: int, verbose: bool=True):
        super().__init__(env)

        self.env = env
        self.frames_per_action = frames_per_action

        self.step_report = 100
        self.total_steps = 0

        self.verbose = verbose


    def slice_obs(self, obs):
        return obs[15:196, 8:]


    def reset(self, seed=None):
        self.total_steps = 0

        obs, info = self.env.reset(seed=seed)

        return self.slice_obs(obs), info
    

    def step(self, action_data):
        i = 0
        reward = 0.0
        while i < self.frames_per_action and reward == 0.0:
            obs, reward, terminated, truncated, info = self.env.step(action_data)
            i += 1
        
        self.total_steps += 1

        return self.slice_obs(obs), reward, terminated, truncated, info
    
class EpisodeTrackerWrapper(AtariWrapper):
    def __init__(self, env: Env, frames_per_action: int, max_events_per_cat: np.array, relevant_cat_count: int=2, verbose: bool=True):
        super().__init__(env, frames_per_action, verbose)

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(12,4), dtype=np.float32)
        self.max_events_per_cat = max_events_per_cat
        self.episode_tracker = EpisodeTracker(np.array([np.array([142, 142, 142]), np.array([170, 170, 170]), np.array([214, 214, 214])]), relevant_cat_count)
        self.to_render = None


    def create_state(self, event: MovementEvent, episode_tracker: EpisodeTracker):
        pos = event.current_pos
        vel = episode_tracker.get_event_vel(event)
        
        return np.array([pos.x / 182.0, pos.y / 152.0, vel.x / 7.75, vel.y / 7.75])
        


    def step(self, action_data):
        obs, reward, terminated, truncated, info = super().step(action_data)
        render, events, categories = self.episode_tracker.process_frame(obs)

        self.to_render = render
        
        return self.create_observation(events, categories), reward, terminated, truncated, info
    

    def create_observation(self, events, categories):
        # Order categories
        for i in range(1, len(categories)):
            key = categories[i]
            j = i - 1

            while j >= 0 and key.size.area() > categories[j].size.area():
                categories[j + 1] = categories[j]
                j -= 1
            categories[j + 1] = key
        cat_max_pos = {}
        cat_first_pos = {}
        cat_pos = {}
        count = 0
        for i in range(len(categories)):
            cat_max_pos[categories[i]] = self.max_events_per_cat[i]
            cat_pos[categories[i]] = 0
            cat_first_pos[categories[i]] = count
            count += self.max_events_per_cat[i]

        # Order events
        for i in range(1, len(events)):
            key = events[i]
            j = i - 1

            while j >= 0 and key.current_pos.y > events[j].current_pos.y:
                events[j + 1] = events[j]
                j -= 1
            events[j + 1] = key

        # Create observation
        obs = np.zeros((np.sum(self.max_events_per_cat), 4))
        for event in events:
            if event.category != "MOVEMENT":
                continue

            cat = event.obj_category
            if cat_pos[cat] >= cat_max_pos[cat]:
                continue

            obs[cat_first_pos[cat] + cat_pos[cat], :] = self.create_state(event, self.episode_tracker)
            
            cat_pos[cat] += 1
        
        return obs


    def reset(self, seed=None):
        obs, info = super().reset(seed)
        self.episode_tracker.finish_episode()
        render, events, categories = self.episode_tracker.process_frame(obs)
        
        self.to_render = render
        return self.create_observation(events, categories), info
    

    def render(self):
        return self.to_render

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = EpisodeTrackerWrapper(env, 1, np.array([10, 2]))

        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(48, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs.flatten()).to(device))
            actions = torch.tensor([torch.argmax(q_values, dim=0)]).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations.flatten(1)).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations.flatten(1)).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
