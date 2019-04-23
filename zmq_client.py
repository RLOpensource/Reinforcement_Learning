import zmq, time, sys
import tensorflow as tf
import dqn
import json
import numpy as np
from utils import NumpyEncoder
from tensorboardX import SummaryWriter
import time
import gym
from wrappers import make_env

def run_client(agent, env_name, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(f'tcp://localhost:{port}')
    memory_json = json.dumps({'message': 'parameter'}, cls=NumpyEncoder)
    sock.send_json(memory_json)
    rep = json.loads(sock.recv_json())
    main_p, target_p = rep['data'][0], rep['data'][1]
    agent.write_parameter(main_p, target_p)

    if env_name in ['MountainCar-v0', 'CartPole-v0']:
        sction = 1
        env = gym.make(env_name)
    elif env_name in ["PongNoFrameskip-v4"]:
        sction = 2
        env = make_env(env_name) 

    step = 0
    done = False
    state = env.reset()
    score = 0
    step_per_q_value = 1
    step_per_loss = 1
    sum_of_q_value = 0
    agent.n_step_buffer.reset()
    episode = 0

    while True:
        step += 1
        action, q_value = agent.get_action(state, agent.epsilon)

        if not q_value == None:
            sum_of_q_value += q_value
            step_per_q_value += 1

        if sction == 1:
            next_state, reward, done, info = env.step(action)
        if sction == 2:
            next_state, reward, done, info = env.step(action + 1)

        score += reward
        agent.append_to_memory(state, next_state, reward, done, action)
        state = next_state

        if step % agent.batch_size == 0:
            minibatch, idxs, IS_weight = agent.memory.sample(agent.batch_size)
            minibatch = np.array(minibatch)
            state_batch = np.stack(minibatch[:, 0])
            next_state_batch = np.stack(minibatch[:, 1])
            discounted_reward_batch = np.stack(minibatch[:, 2])
            done_batch = np.stack(minibatch[:, 3])
            action_batch = np.stack(minibatch[:, 4])
            send_data = {
                'state': state_batch,
                'next_state': next_state_batch,
                'reward': discounted_reward_batch,
                'done': done_batch,
                'action': action_batch
            }
            memory_json = json.dumps({'message': 'data', 'minibatch': send_data}, cls=NumpyEncoder)
            sock.send_json(memory_json)
            json.loads(sock.recv_json())

            q_value_for_select_action = agent.sess.run(agent.main, feed_dict={agent.x_ph: next_state_batch})
            selected_action = np.argmax(q_value_for_select_action, axis=1)
            target_q_value = agent.sess.run(agent.target, feed_dict={agent.x_ph: next_state_batch})
            target_value = [np.power(agent.gamma, agent.n_step) * q[a] * (1-int(d)) for a, q, d in zip(selected_action, target_q_value, done_batch)]
            target = np.stack(discounted_reward_batch) + np.stack(target_value)

            state_value = agent.sess.run(agent.main, feed_dict={agent.x_ph: state_batch})
            state_value = [sv[a] for a, sv in zip(action_batch, state_value)]
            td_error = np.abs(target - np.stack(state_value))

            for i in range(agent.batch_size):
                idx = idxs[i]
                agent.memory.update(idx, td_error[i])

        if step % agent.update_size == 0:
            memory_json = json.dumps({'message': 'parameter'}, cls=NumpyEncoder)
            sock.send_json(memory_json)
            rep = json.loads(sock.recv_json())
            main_p, target_p = rep['data'][0], rep['data'][1]
            agent.write_parameter(main_p, target_p)

        if done:
            episode += 1
            print(episode, score, sum_of_q_value / step_per_q_value, agent.epsilon)
            score = 0
            state = env.reset()
            done = False
            sum_of_q_value = 0
            step_per_q_value = 1
            agent.epsilon = 1 / (episode * 0.1 + 1)
            agent.n_step_buffer.reset()


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
        train_size=1,
        update_size=100,
        activation=tf.nn.relu
    )
    run_client(agent, env, 5555)