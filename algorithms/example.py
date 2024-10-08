import yaml
from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.evaluator import run_episode
from pogema_toolbox.create_env import Environment
from pypibt.inference import PIBTInference, PIBTInferenceConfig
from create_env import create_env_base

def main():
    with open('experiments/01-random/maps.yaml', 'r') as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)
    ToolboxRegistry.register_env('Environment', create_env_base, Environment)
    algo_cfg = PIBTInferenceConfig(device='cpu', num_process=8, centralized=True)
    env_cfg = Environment(num_agents=30, map_name='validation-random-seed-000', max_episode_steps=100, seed=41, observation_type='MAPF', collision_system='soft', with_animation=True, on_target='nothing')
    env = ToolboxRegistry.create_env('Environment', **env_cfg.dict())
    algo = PIBTInference(algo_cfg)
    print(run_episode(env, algo))
    env.save_animation('out.svg')


if __name__ == '__main__':
    main()
