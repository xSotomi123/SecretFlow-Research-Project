# SecretFlow Experiment Source Code

Source code used in my research project to assess the integration complexity and usability of [SecretFlow](https://github.com/secretflow/secretflow).

## Installation guide:

This framework was installed by following SecretFlow's [official installation guide](https://www.secretflow.org.cn/en/docs/secretflow/v1.12.0b0/getting_started/installation#installation).
First create a clean virtual environment:

```
conda create -n sf python=3.10
```
```
conda activate sf
```

Now install SecretFlow:

```
pip install -U secretflow
```

Now you can run the `installation_tutorial_test.py` file to see if SecretFlow has been successfully installed.

The tutorial code from their [deployment guide](https://www.secretflow.org.cn/en/docs/secretflow/v1.12.0b0/getting_started/deployment#deployment) was also executed and can be found in the `deployment_tutorial_test.py` file (do note that you need to change the `ip:port` found on line 4 to your respective ip and port of head node). Before running the code, you need to execute the following commands in the terminal (change `ip` to your ip and `port` to any desired free port, for the second command make sure that the `ip:port` is your ip followed by the port chosen for the head node):

```
ray start --head --node-ip-address="ip" --port="port" --resources='{"alice": 16}' --include-dashboard=False --disable-usage-stats
```
```
ray start --address="ip:port" --resources='{"bob": 16}' --disable-usage-stats
```
Now you can run the code in the file. To shut down the cluster, execute:

```
ray stop
```

In `mix_fl_lr_tutorial_test.py` you can find the code from their [mix federated learning - logistic regression tutorial](https://www.secretflow.org.cn/en/docs/secretflow/v1.12.0b0/tutorial/mix_lr) and it can be ran without executing any additional terminal commands if you've successfully installed the framework.
