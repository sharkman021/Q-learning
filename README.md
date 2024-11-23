poetry shellで仮想環境に入る
あるいは
emulate bash -c '. /Users/kohei/semi_q/.venv/bin/activate'

TensorBoardの使用

TensorBoardを使用して学習の進行状況をリアルタイムでモニタリングすることができます。以下のコマンドでTensorBoardを起動し、ブラウザで確認します。

tensorboard --logdir=runs
ブラウザで表示されるURL（通常は http://localhost:6006/）にアクセスし、グラフやスカラー値を確認します。

main.py:

プロジェクトのエントリーポイント。
コマンドライン引数を解析し、環境 (ElevatorEnv) とエージェント (QLearningAgent) を初期化します。
トレーニングや評価のループを実行し、Q テーブルを更新・保存します。
environment.py と agent.py をインポートして使用します。
environment.py:

エレベータシミュレーション環境を定義。
Passenger、Elevator、ElevatorEnv クラスを提供。
エレベータの動作や乗客の生成・移動を管理します。
main.py からインスタンス化され、シミュレーションを進行させます。
agent.py:

Q 学習エージェントを定義。
main.py からインスタンス化され、エージェントの行動選択や Q テーブルの更新を行います。
学習率、割引率、探索率などのハイパーパラメータを管理します。
utils.py:

ユーティリティ関数を提供。
状態や方向のエンコーディングに使用される可能性がありますが、現在は未使用。
将来的な拡張や他のモジュールでの利用を想定。

学習の実行
poetry run python main.py --config config.yaml --episodes 1000 --save_q q_table.npy --render
poetry run python Q-learning/main.py --config config.yaml --episodes 1000 --save_q q_table.npy --render


評価コマンドの実行
poetry run python main.py --config config.yaml --episodes 1 --load_q q_table.npy --evaluate --render

a. 基本的なトレーニングの実行

poetry run python main.py --config config.yaml --episodes 1000 --save_q q_table.npy
オプションの説明:

--config config.yaml: 環境設定ファイルを指定します。デフォルトは config.yaml ですが、他の設定ファイルを使用する場合はパスを指定します。
--episodes 1000: トレーニングするエピソード数を指定します。ここでは 1000 エピソードを実行します。
--save_q q_table.npy: 学習後に Q テーブルを保存するファイル名を指定します。デフォルトは q_table.npy です。
例:


poetry run python main.py --config config.yaml --episodes 5000 --save_q trained_q_table.npy
b. 追加オプションの使用
i. 最大ステップ数の変更
各エピソードで実行する最大ステップ数を変更することができます。デフォルトは 5400 ステップです。


poetry run python main.py --config config.yaml --episodes 1000 --max_steps 10000 --save_q q_table.npy
--max_steps 10000: 各エピソードで最大 10000 ステップまで実行します。
ii. 環境のレンダリング
シミュレーションの状態をリアルタイムで視覚的に確認したい場合、--render オプションを追加します。ただし、ステップ数が多い場合は出力が大量になるため、トレーニングの進行が遅くなる可能性があります。


poetry run python main.py --config config.yaml --episodes 10 --render --save_q q_table.npy
--render: シミュレーションの状態をコンソールに表示します。
注意: 大量のエピソードでレンダリングを有効にすると、実行速度が低下するため、必要な場合のみ使用してください。


poetry run python main.py --config config.yaml
poetry run python Q-learning/main.py --config Q-learning/config.yaml
