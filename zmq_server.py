import zmq, time
import tensorflow as tf
import dqn
import json
from utils import NumpyEncoder

def run_server(agent, env, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind('tcp://*:' + str(port))
    step = 0
    while True:
        step += 1
        memory = json.loads(sock.recv_json())

        if memory['message'] == 'parameter':
            parameter_json = json.dumps({'message': 'parameter', 'data': agent.get_parameter()}, cls=NumpyEncoder)
            sock.send_json(parameter_json)
        if memory['message'] == 'data':
            batch_data = memory['minibatch']
            state = batch_data['state']
            next_state = batch_data['next_state']
            reward = batch_data['reward']
            done = batch_data['done']
            action = batch_data['action']

            for s, ns, r, d, a in zip(state, next_state, reward, done, action):
                agent.append_to_memory(s, ns, r, d, a)
            parameter_json = json.dumps({'message': 'parameter', 'data': agent.get_parameter()}, cls=NumpyEncoder)
            sock.send_json(parameter_json)

        if step % agent.update_size == 0:
            agent.update()
            agent.memory.reset()

if __name__ == '__main__':
    env = 'MountainCar-v0'
    agent = dqn.DQN(
        max_length=1e4,
        state_size=2,
        output_size=3,
        hidden=[256, 256],
        n_step=10,
        batch_size=64,
        gamma=0.999,
        lr=0.001,
        train_size=100,
        update_size=100,
        activation=tf.nn.relu
    )
    run_server(agent, env, 5555)