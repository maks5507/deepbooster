# DeepBooster: Colab-driven multi-GPU training of large Neural Nets
**Authors:** Maksim Eremeev (@maks5507, me@maksimeremeev.com), Mikhail Khramenkov (@0x2500)

## Installation

```bash
python setup.py build
pip install .
```

## Usage:

Below is a sample script can be run in multiple colab notebooks to test the system. Make sure you specify different `my_queue` and `ignition_queue` for different instances.

```python
! git clone -b dev https://github.com/maks5507/amqp-interface.git; cd amqp-interface; python3 setup.py build; pip install .
  
! git clone https://github.com/maks5507/deepbooster.git; cd deepbooster; python3 setup.py build; pip install .
  
import torch
import torch.nn as nn

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 1),
            nn.Tanh()
        )

    def forward(self, x):
        import time
        time.sleep(10)
        return self.layers(x)  
      
import random
import json


path = '.'
with open(f'{path}/chunk1.jsonl', 'w') as f:
    for i in range(int(5)):
        curr = json.dumps({'input': torch.rand((1000, 10)).tolist(), 'label': random.randint(0, 4)})
        f.write(f'{curr}\n')
        

import torch.nn.functional as F
import deepbooster

model = Test()

trainer_params = {
    'model': model,
    'criterion': nn.MSELoss(),
    'optimizer': torch.optim.Adam(model.parameters(), lr=0.01),

    'device': torch.device('cuda'),
    'n_epoch': 1000,

    'ignition_queue': 'start_1',
    'my_queue': 'test1',
    'trainers_queues': ['test1', 'test2'],

    'url_parameters': 'amqp://user:pass@35.222.31.138:5672',

    'chunk_path': './chunk1.jsonl',
    'transformer': torch.Tensor,
    'apply_transformer_to_label': True
}

trainer = deepbooster.Trainer(**trainer_params)

trainer.start()
```

The process will be blocked unless you post any message to the ignition queue of your instance.

## Message Queue

The RabbitMQ is used for sync of the workers. As the computed gradients are passed through the queue on each step, make sure the RAM & Disk do not overflow. The RMQ bandwidth is enough to carry a gradient of > 1.4B parameter network.

As the queue should be accessible from any colab instance, currently we use a pubic server with open 5672 and 15672 ports: `35.222.31.138`.

The RMQ needs to be setup (convenient scripts to follow) as follows:

* Create an ignition queue and sync queue for each instance you want to launch
* Bind both queues to the `amq.topic` exchange
* Purge both ignition and sync queues before running a new training procedure

