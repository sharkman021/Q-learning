import yaml
import argparse
import numpy as np
from lifcon import World, make_person_from_jsval, ControllerStatus, Wait, Move
from lifcon_dqn import LiftControllerDQN
import gzip
import logging
import json
from fractions import Fraction

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd
import itertools

logging.basicConfig(level=logging.DEBUG)

#mxnetを用いてニューラルネットワークの構築とトレーニング

parser = argparse.ArgumentParser()
#世界設定ファイルのパス→環境？
parser.add_argument('--world', type=str,
                    help='world configuration')
#リプレイファイルの読み込みのパス
parser.add_argument('--replay', type=str, default=None,
                    help='Read replay file')
#リプレイリストのパスの読み込みのパス
parser.add_argument('--replaylist', type=str, default=None,
                    help='Read replay file')
#DQNパラメータの読み込み
parser.add_argument('--loadparam', type=str, default=None,
                    help='Read DQN parameter')
#DQNパラメータの保存パス
parser.add_argument('--saveparam', type=str, default=None,
                    help='Write DQN parameter')
#seed
parser.add_argument('--seed', type=int, default=0,
                    help='seed')
#バッチサイズ→一度に入力するデータの数
parser.add_argument('--batchsize', type=int, default=128,
                    help='batchsize')
#割引率？？
parser.add_argument('--gamma', type=float, default=0.9,
                    help='gamma')
#割引率？？
parser.add_argument('--lr', type=float, default=0.0002,
                    help='lr')
#許容誤差
parser.add_argument('--tolerance', type=float, default=0.05,
                    help='tolerance')
#最大エポック数
parser.add_argument('--maxnepoch', type=int, default=10,
                    help='nepoch')
#double_DQN
parser.add_argument('--double-dqn', action='store_true',
                    default=False, help='Use double DQN')


opt = parser.parse_args()

#yamlファイルを読み込んで環境を構築
world_conf = yaml.load(open(opt.world).read())
world = World(world_conf)

mx.random.seed(opt.seed)

ctx = mx.cpu()

#LiftControllerDQNとリファレンスモデルを初期化して必要に応じて既存のパラメータをロード
dqn = LiftControllerDQN(world.nlifts, world.nfloors, world.lift_inv_speed)
#ターゲットネットワークとして使用している→トレーニングの安定化に寄与？
dqn_ref = LiftControllerDQN(world.nlifts, world.nfloors, world.lift_inv_speed)

dqn.load_params(opt.loadparam, ctx=ctx)
dqn_ref.load_params(opt.loadparam, ctx=ctx)


#リプレイデータの読み込み
#指定されたリプレイファイルを読み込み、replayリストに統合
replayfiles = []
if opt.replay is not None:
    replayfiles.append(opt.replay)
if opt.replaylist is not None:
    for l in open(opt.replaylist):
        l = l.strip()
        if l.startswith('#') or len(l) == 0:
            continue
        replayfiles.append(l)

replay = []
for replayfile in replayfiles:
    logging.info("Use replay file: %s", replayfile)
    rep = json.load(gzip.open(replayfile, "r"))
    replay.extend(rep)
    replay.append(None)

logging.info("#Ticks=%d", len(replay) - 1)

all_idxs = list(range(len(replay)-1))

#損失関数とトレーナーの設定
#損失関数として二乗誤差を使用
#評価指標として平均絶対誤差を使用
loss = gluon.loss.L2Loss()

def status_from_jsval(jv):
    return ControllerStatus(
        wait_up=[n > 0 for n in jv['wait_up']],
        wait_down=[n > 0 for n in jv['wait_dn']],
        locations=[Fraction(num, den) for num, den in jv['locations']],
        members=[[make_person_from_jsval(p) for p in ps]
                 for ps in jv['inlift_p']],
        status=jv['statuses']
    )

metric = mx.metric.MAE()
#トレーナーにはAdamオプティマイザーを使用して学習率を指定？？？？？？
#Adamオプティマイザーとは
trainer = gluon.Trainer(dqn.collect_params(),
                        'adam', {'learning_rate': opt.lr})


#トレーニングループ
batchnum = 0
nepoch = 0
#外側ループ、指定された最大エポック数(opt.maxnepoch)、または許容誤差(opt.tolerance)に達するまでトレーニングを継続
#parserで上記は指定している
while True:
    np.random.shuffle(all_idxs)
    idxit = iter(all_idxs)
    metric.reset()

    #バッチサイズ分のインデックスを取得、対処するリプレイデータを処理
    while True:
        idxs = list(itertools.islice(idxit, opt.batchsize))

        if len(idxs) < opt.batchsize:
            break

        cur_lift = np.zeros((opt.batchsize, dqn.nlifts, dqn.nliftinfo))
        cur_side = np.zeros((opt.batchsize, dqn.nsideinfo))
        nxt_lift = np.zeros((opt.batchsize, dqn.nlifts, dqn.nliftinfo))
        nxt_side = np.zeros((opt.batchsize, dqn.nsideinfo))

        #アクション選択機を生成
        cur_Asel = np.zeros((opt.batchsize, dqn.nlifts, dqn.nactions))
        
        #報酬をリプレイデータから取得
        rewards = np.zeros((opt.batchsize, 1))

        for i, idx in enumerate(idxs):
            cur = replay[idx]
            nxt = replay[idx + 1]

            if cur is None:
                continue

            cur_stat = status_from_jsval(cur)

            lift, side = dqn.encode_state(cur_stat, cur['tick'])
            cur_lift[i, :, :] = lift
            cur_side[i, :] = side

            goals = []
            for v in cur['goals']:
                if v == "STOP":
                    goals.append(Wait())
                else:
                    goals.append(Move(dest=int(v)))

            cur_Asel[i, :, :] = dqn.make_action_selector(cur_stat,
                                                         goals, cur['acceptst'])

            if nxt is None:
                continue

            assert cur['tick'] + 1 == nxt['tick']

            lift, side = dqn.encode_state(status_from_jsval(nxt), nxt['tick'])
            nxt_lift[i, :, :] = lift
            nxt_side[i, :] = side

            aux_reward = 0.0

            # rewards in replay are computed before making an action
            rewards[i, 0] = sum(nxt['rewards']) + aux_reward


        cur_side = mx.nd.array(cur_side)
        cur_Asel = mx.nd.array(cur_Asel)
        rewards = mx.nd.array(rewards)

        #ターゲットQ値の計算。ターゲットネットワーク(dqn_ref)を用いて次の状態のQ値を計算。
        nxt_Q = dqn_ref(mx.nd.array(nxt_lift), mx.nd.array(nxt_side))

        #ダブルDQNを使用する場合DQNを用いてアクションを選択してQ値を評価
        if opt.double_dqn:
            # In double-DQN, action is evaluated by the online (current) policy
            # but uses Q-values estimated by the offline policy
            nxt_Q_from_cur = dqn(mx.nd.array(nxt_lift), mx.nd.array(nxt_side))

        nxt_Asel = np.zeros((opt.batchsize, dqn.nlifts, dqn.nactions))
        #損失関数とバックプロぱゲーしょん
        #現在のQ値（qs）とターゲットQ値（target）との二乗誤差を計算。勾配を計算しパラメータを更新
        for i, idx in enumerate(idxs):
            cur = replay[idx]
            nxt = replay[idx + 1]
            if cur is None or nxt is None:
                continue # nxt_Asel stays at zero, thus Q values will be zero

            nxt_stat = status_from_jsval(nxt)

            if opt.double_dqn:
                goals, flags = dqn.make_action(nxt_Q_from_cur.asnumpy()[i, :, :],
                                               nxt_stat, 0)
            else:
                goals, flags = dqn.make_action(nxt_Q.asnumpy()[i, :, :],
                                               nxt_stat, 0)
            nxt_Asel[i, :] = dqn.make_action_selector(nxt_stat, goals, flags)
        nxt_Asel = mx.nd.array(nxt_Asel)

        target = rewards + opt.gamma * mx.nd.sum(nxt_Q * nxt_Asel, axis=2).reshape((opt.batchsize, dqn.nlifts))

        with autograd.record():
            qs = dqn(mx.nd.array(cur_lift), mx.nd.array(cur_side))
            qa_s = (cur_Asel * qs).sum(axis=2).reshape((opt.batchsize, dqn.nlifts))

            err = loss(qa_s, target)

            err.backward()
            metric.update([target,], [qa_s,])
        
        #評価指数の更新とログの出力
        _, mae = metric.get()
        logging.info('[%07d] MAE = %f' % (batchnum, mae))
        batchnum += 1

        trainer.step(opt.batchsize)

    dqn.save_params(opt.saveparam)
    logging.info('Total MAE = %f' % (mae,))

    _, mae = metric.get()
    if mae < opt.tolerance:
        break
    nepoch += 1

    if nepoch >= opt.maxnepoch:
        break