import spu
import shutil
import logging
import tempfile
import numpy as np
import secretflow as sf
from typing import List
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from secretflow.ml.linear import FlLogisticRegressionMix
from secretflow.security.aggregation import SecureAggregator

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running SecretFlow runtime already.
# sf.shutdown()

sf.init(['alice', 'bob', 'carol', 'dave', 'eric'], address='local', num_cpus=64)

alice, bob, carol, dave, eric = (
    sf.PYU('alice'),
    sf.PYU('bob'),
    sf.PYU('carol'),
    sf.PYU('dave'),
    sf.PYU('eric'),
)

features, label = load_breast_cancer(return_X_y=True, as_frame=True)
features.iloc[:, :] = StandardScaler().fit_transform(features)
label = label.to_frame()

feat_list = [
    features.iloc[:, :10],
    features.iloc[:, 10:20],
    features.iloc[:, 20:],
]

alice_y0, alice_y1, alice_y2 = label.iloc[0:200], label.iloc[200:400], label.iloc[400:]
alice_x0, alice_x1, alice_x2 = (
    feat_list[0].iloc[0:200, :],
    feat_list[0].iloc[200:400, :],
    feat_list[0].iloc[400:, :],
)
bob_x0, bob_x1, bob_x2 = (
    feat_list[1].iloc[0:200, :],
    feat_list[1].iloc[200:400, :],
    feat_list[1].iloc[400:, :],
)
carol_x, dave_x, eric_x = (
    feat_list[2].iloc[0:200, :],
    feat_list[2].iloc[200:400, :],
    feat_list[2].iloc[400:, :],
)

tmp_dir = tempfile.mkdtemp()

def filepath(filename):
    return f'{tmp_dir}/{filename}'

alice_y0_file, alice_y1_file, alice_y2_file = (
    filepath('alice_y0'),
    filepath('alice_y1'),
    filepath('alice_y2'),
)
alice_x0_file, alice_x1_file, alice_x2_file = (
    filepath('alice_x0'),
    filepath('alice_x1'),
    filepath('alice_x2'),
)
bob_x0_file, bob_x1_file, bob_x2_file = (
    filepath('bob_x0'),
    filepath('bob_x1'),
    filepath('bob_x2'),
)
carol_x_file, dave_x_file, eric_x_file = (
    filepath('carol_x'),
    filepath('dave_x'),
    filepath('eric_x'),
)

alice_x0.to_csv(alice_x0_file, index=False)
alice_x1.to_csv(alice_x1_file, index=False)
alice_x2.to_csv(alice_x2_file, index=False)
bob_x0.to_csv(bob_x0_file, index=False)
bob_x1.to_csv(bob_x1_file, index=False)
bob_x2.to_csv(bob_x2_file, index=False)
carol_x.to_csv(carol_x_file, index=False)
dave_x.to_csv(dave_x_file, index=False)
eric_x.to_csv(eric_x_file, index=False)
alice_y0.to_csv(alice_y0_file, index=False)
alice_y1.to_csv(alice_y1_file, index=False)
alice_y2.to_csv(alice_y2_file, index=False)

vdf_x0 = sf.data.vertical.read_csv(
    {alice: alice_x0_file, bob: bob_x0_file, carol: carol_x_file}
)
vdf_x1 = sf.data.vertical.read_csv(
    {alice: alice_x1_file, bob: bob_x1_file, dave: dave_x_file}
)
vdf_x2 = sf.data.vertical.read_csv(
    {alice: alice_x2_file, bob: bob_x2_file, eric: eric_x_file}
)
vdf_y0 = sf.data.vertical.read_csv({alice: alice_y0_file})
vdf_y1 = sf.data.vertical.read_csv({alice: alice_y1_file})
vdf_y2 = sf.data.vertical.read_csv({alice: alice_y2_file})

x = sf.data.mix.MixDataFrame(partitions=[vdf_x0, vdf_x1, vdf_x2])
y = sf.data.mix.MixDataFrame(partitions=[vdf_y0, vdf_y1, vdf_y2])

def heu_config(sk_keeper: str, evaluators: List[str]):
    return {
        'sk_keeper': {'party': sk_keeper},
        'evaluators': [{'party': evaluator} for evaluator in evaluators],
        'mode': 'PHEU',
        'he_parameters': {
            'schema': 'paillier',
            'key_pair': {'generate': {'bit_size': 2048}},
        },
    }

heu0 = sf.HEU(heu_config('alice', ['bob', 'carol']), spu.spu_pb2.FM128)
heu1 = sf.HEU(heu_config('alice', ['bob', 'dave']), spu.spu_pb2.FM128)
heu2 = sf.HEU(heu_config('alice', ['bob', 'eric']), spu.spu_pb2.FM128)
aggregator0 = SecureAggregator(alice, [alice, bob, carol])
aggregator1 = SecureAggregator(alice, [alice, bob, dave])
aggregator2 = SecureAggregator(alice, [alice, bob, eric])

logging.root.setLevel(level=logging.INFO)

model = FlLogisticRegressionMix()

model.fit(
    x,
    y,
    batch_size=64,
    epochs=3,
    learning_rate=0.1,
    aggregators=[aggregator0, aggregator1, aggregator2],
    heus=[heu0, heu1, heu2],
)

y_pred = np.concatenate(sf.reveal(model.predict(x)))

auc = roc_auc_score(label.values, y_pred)
acc = np.mean((y_pred > 0.5) == label.values)
print('auc:', auc, ', acc:', acc)

shutil.rmtree(tmp_dir, ignore_errors=True)