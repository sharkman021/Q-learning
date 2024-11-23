# main.py

import yaml
from environment import ElevatorEnv
from agent import QLearningAgent
import argparse
import numpy as np
import mxnet as mx
import logging
from torch.utils.tensorboard import SummaryWriter  # TensorBoardを使用する場合

def parse_args():
    parser = argparse.ArgumentParser()
    # 設定ファイルのパスを設定→config.yamlに記載した
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config.yaml')
    # トレーニングの数。1000で設定
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    # エピソードの最大ステップ。5400で設定
    parser.add_argument('--max_steps', type=int, default=5400,
                        help='Maximum steps per episode')
    # 学習後のファイル名。q_table.paramsで設定
    parser.add_argument('--save_q', type=str, default='q_table.params',
                        help='File to save Q-table')
    # Qテーブルをロードする時のファイル名。デフォルトなし。
    parser.add_argument('--load_q', type=str, default=None,
                        help='File to load Q-table')
    # 環境をレンダリングするオプション
    parser.add_argument('--render', action='store_true',
                        help='Render environment')
    # 評価モードを有効にするオプション
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate the agent using the loaded Q-table without exploration')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # ハイパーパラメータの適用
    hyper = config['hyperparameters']
    learning_rate = hyper['learning_rate']
    gamma = hyper['gamma']
    epsilon = hyper['epsilon']
    epsilon_min = hyper['epsilon_min']
    epsilon_decay = hyper['epsilon_decay']
    batch_size = hyper['batch_size']
    max_episodes = hyper['max_episodes']
    loss_function = hyper['loss_function']
    optimizer = hyper['optimizer']

    # ロギングの設定
    logging.basicConfig(level=logging.INFO, filename='training.log',
                        format='%(asctime)s %(levelname)s:%(message)s')

    # TensorBoardの設定
    writer = SummaryWriter(log_dir='runs/experiment_1')

    # 環境の初期化
    env = ElevatorEnv(config)

    # Define action space
    action_space = ['up', 'down', 'idle']

    # Initialize Q-Learning Agent
    agent = QLearningAgent(action_space=action_space,
                           alpha=learning_rate, gamma=gamma,
                           epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                           loss_function=loss_function, optimizer=optimizer, learning_rate=learning_rate)

    # --load_qオプションが指定されていると保存されているQテーブルをロード
    if args.load_q:
        agent.load_q_table(args.load_q)
        print(f"Loaded Q-table from {args.load_q}")
        # --evaluateオプションが指定されていると完全に学習済みのポリシーを使用
        if args.evaluate:
            agent.epsilon = 0.0  # 評価モードでは探索を無効化
            print("Evaluation mode: epsilon set to 0.0 (no exploration)")

    rewards_history = []  # 報酬履歴のリストを初期化

    # エピソードの実行
    for episode in range(1, max_episodes + 1):
        env.reset()
        state = env._get_state()
        total_rewards = 0
        total_loss = 0.0  # 損失の合計を初期化
        for step in range(args.max_steps):
            actions = []
            for eid in range(env.num_lifts):
                # 状態を文字列としてQテーブルのキーに使用
                key = str(state)
                action = agent.get_action(key)
                actions.append(action_space[action])
            next_state, reward, done, total_waiting = env.step(actions)
            loss = agent.update_q(str(state), actions.index(actions[0]), reward, str(next_state), done)
            state = next_state
            total_rewards += reward
            total_loss += loss
            if args.render:
                env.render()
            if done:
                break
        if not args.evaluate:
            agent.decay_epsilon()
        rewards_history.append(total_rewards)  # 報酬を履歴に追加
        logging.info(f"Episode {episode}: Total Reward: {total_rewards}, Epsilon: {agent.epsilon:.4f}, Loss: {total_loss:.4f}")
        writer.add_scalar('Total Reward', total_rewards, episode)
        writer.add_scalar('Epsilon', agent.epsilon, episode)
        writer.add_scalar('Loss', total_loss, episode)

        # 定期的にTensorBoardに書き込む
        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_rewards}, Epsilon: {agent.epsilon:.4f}, Loss: {total_loss:.4f}")

    if not args.evaluate:
        # トレーニングが完了したらQテーブルを保存
        agent.save_q_table(args.save_q)
        print(f"Q-table saved to {args.save_q}")

    # トレーニング終了後
    writer.close()

    # 報酬履歴を保存
    np.save('rewards_history.npy', rewards_history)
    print("Rewards history saved to rewards_history.npy")

if __name__ == "__main__":
    main()
