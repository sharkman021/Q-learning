# agent.py

import mxnet as mx
from mxnet import gluon
import numpy as np
import json

class QLearningAgent:
    def __init__(self, action_space, alpha, gamma, epsilon, epsilon_min, epsilon_decay, loss_function, optimizer, learning_rate):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Qテーブルの初期化（ニューラルネットワークを使用）
        self.q_table = {}  # 状態-アクションのQ値を格納（未使用）

        # 損失関数の設定
        if loss_function == 'huber':
            self.loss = gluon.loss.HuberLoss()
        elif loss_function == 'l2':
            self.loss = gluon.loss.L2Loss()
        else:
            raise ValueError(f"Unsupported loss function type: {loss_function}")

        # モデルの定義（単純なニューラルネットワーク）
        self.model = gluon.nn.Sequential()
        with self.model.name_scope():
            self.model.add(gluon.nn.Dense(128, activation='relu'))
            self.model.add(gluon.nn.Dense(len(action_space)))
        self.model.initialize(mx.init.Xavier())

        # オプティマイザーの設定
        if optimizer == 'adam':
            self.trainer = gluon.Trainer(self.model.collect_params(),
                                        'adam',
                                        {'learning_rate': learning_rate})
        elif optimizer == 'sgd':
            self.trainer = gluon.Trainer(self.model.collect_params(),
                                        'sgd',
                                        {'learning_rate': learning_rate})
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer}")

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.action_space))
        else:
            state_encoded = self.encode_state(state)
            q_values = self.model(mx.nd.array([state_encoded]))
            return int(mx.nd.argmax(q_values, axis=1).asscalar())

    def update_q(self, state, action, reward, next_state, done):
        state_encoded = self.encode_state(state)
        next_state_encoded = self.encode_state(next_state)

        # Q値の予測
        with mx.autograd.record():
            q_values = self.model(mx.nd.array([state_encoded]))
            q_value = q_values[0, action]

            # ターゲットの計算
            if done:
                target = mx.nd.array([reward])
            else:
                next_q_values = self.model(mx.nd.array([next_state_encoded]))
                target = reward + self.gamma * mx.nd.max(next_q_values)

            # 損失の計算
            loss = self.loss(q_value, target)

        # 損失のバックプロパゲーションとパラメータ更新
        loss.backward()
        self.trainer.step(1)

        # 損失を返す（ロギング用）
        return loss.mean().asscalar()

    def encode_state(self, state):
        # 状態を数値ベクトルにエンコード（例として簡単な方法）
        # 実際のエンコード方法はプロジェクトに応じて調整
        return np.array(state, dtype=np.float32)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_q_table(self, filename):
        # モデルのパラメータを保存
        self.model.save_parameters(filename)

    def load_q_table(self, filename):
        # モデルのパラメータをロード
        self.model.load_parameters(filename, ctx=mx.cpu())
