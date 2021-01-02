#
# Created by maks5507 (me@maksimeremeev.com)
#

from typing import List
from pathlib import Path
import rmq_interface


class Launcher:
    def __init__(self, rmq_url_parameters: str, ignition_queues: List[str]):
        self.rmq_interface_parameters = rmq_url_parameters
        self.ignition_queues = ignition_queues
        self.num_workers = len(self.ignition_queues)

    def connect(self):
        self.interface = rmq_interface.RabbitMQInterface(url_parameters=self.rmq_interface_parameters)

    @staticmethod
    def __split_collection(data_file: str, n_jobs: int):
        data_directory = Path(data_file).parent
        chunks = [f'{data_directory}/{i}.chunk' for i in range(n_jobs)]

        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                with open(chunks[i % n_jobs], 'a') as ff:
                    ff.write(line)
        return chunks

    def launch(self, data_path: str):
        chunks = self.__split_collection(data_path, self.num_workers)
        for queue, chumk in zip(self.ignition_queues, chunks):
            self.connect()
            self.interface.publish(routing_key=queue, body=chumk)
