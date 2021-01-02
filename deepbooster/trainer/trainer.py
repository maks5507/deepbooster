#
# Created by maks5507 (me@maksimeremeev.com)
#

from typing import List, Callable
from tqdm.auto import tqdm
import torch
from torch import nn
import rmq_interface
import pickle

import pathmagic
pathmagic.add_to_path(1)
from dataset import Dataset


class Trainer:
    def __init__(self, trainers_queues: List[str], my_queue: str, ignition_queue: str, url_parameters: str,
                 chunk_path: str, model: nn.Module, criterion, optimizer, device: str, n_epoch: int,
                 transformer: Callable = None, apply_transformer_to_label: bool = False):
        self.trainers_queues = trainers_queues
        self.my_queue = my_queue
        self.ignition_queue = ignition_queue
        self.n_trainers = len(self.trainers_queues)
        self.url_parameters = url_parameters

        self.dataset = Dataset(chunk_path, transformer, apply_transformer_to_label)

        self.device = device

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epoch = n_epoch

        self.model = self.model.to(self.device)

        self.connect()

    def connect(self):
        self.interface = rmq_interface.RabbitMQInterface(url_parameters=self.url_parameters)

    def start(self):
        self.connect()

        @rmq_interface.consumer_function
        def launch_training(body, props):
            self.train()

        self.interface.get_n_messages(self.ignition_queue, 1, launch_training)

    def train(self):
        self.train_losses = []

        for t in tqdm(range(self.n_epoch)):
            self.model.train()

            for batch_idx, (x, y) in enumerate(self.dataset):
                x = x.to(self.device)
                y = y.to(self.device)

                output = self.model(x)

                loss = self.criterion(output, y)
                self.train_losses += [loss.detach().cpu()]

                self.optimizer.zero_grad()
                loss.backward()

                grads = [layer.grad.detach().cpu().numpy() for layer in self.model.parameters()]

                self.connect()
                for queue in self.trainers_queues:
                    if queue != self.my_queue:
                        self.interface.publish(routing_key=queue, body=pickle.dumps(grads))

                @rmq_interface.consumer_function
                def consumer(body, props):
                    for param, grad in zip(self.model.parameters(), pickle.loads(body)):
                        param.grad += torch.Tensor(grad).to(self.device)

                self.connect()
                self.interface.get_n_messages(self.my_queue, self.n_trainers - 1, consumer)

                for layer in self.model.parameters():
                    layer.grad /= self.n_trainers

                self.optimizer.step()
