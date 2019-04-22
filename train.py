import tensorflow as tf
import gym
from wrappers import make_env
from tensorboardX import SummaryWriter
import dqn
import time

def test(agent, env_name):
    if env_name in ['MountainCar-v0', 'CartPole-v0']:
        sction = 1
        env = gym.make(env_name)
    elif env_name in ["PongNoFrameskip-v4"]:
        sction = 2
        env = make_env(env_name)
    agent.load_model('model/model')
    while True:
        state = env.reset()
        done = False
        while not done:
            env.render()
            time.sleep(0.01)
            action, q_value = agent.get_action(state, 0)
            if sction == 1:
                next_state, reward, done, _ = env.step(action)
            elif sction == 2:
                next_state, reward, done, _ = env.step(action+1)

            state = next_state

def train(agent, env_name):
    writer = SummaryWriter()

    if env_name in ['MountainCar-v0', 'CartPole-v0']:
        sction = 1
        env = gym.make(env_name)
    elif env_name in ["PongNoFrameskip-v4"]:
        sction = 2
        env = make_env(env_name)
    
    step = 0

    for i in range(99999):
        agent.epsilon = 1 / (i * 0.1 + 1)
        done = False
        state = env.reset()
        score = 0
        step_per_q_value = 0
        step_per_loss = 0
        sum_of_q_value = 0
        agent.n_step_buffer.reset()
        agent.save_model('model/model')
        total_loss = 0
        while not done:
            step += 1

            if i % 10 == 0:
                env.render()
            action, q_value = agent.get_action(state, agent.epsilon)
            if not q_value == None:
                sum_of_q_value += q_value
                step_per_q_value += 1

            if sction == 1:
                next_state, reward, done, info = env.step(action)
            elif sction == 2:
                next_state, reward, done, info = env.step(action+1)                    

            score += reward
            agent.append_to_memory(state, next_state, reward, done, action)
            state = next_state
    
            if step > agent.batch_size:
                if step % agent.train_size == 0:
                    step_per_loss += 1
                    loss = agent.update()
                    total_loss += loss
                if step % agent.update_size == 0:
                    agent.update_parameter()
        writer.add_scalar('data/step', step, i)
        writer.add_scalar('data/score', score, i)
        writer.add_scalar('data/epsilon', agent.epsilon, i)
        
        if not step_per_q_value == 0:
            writer.add_scalar('data/average_of_q_value', sum_of_q_value / step_per_q_value, i)
        if not step_per_loss == 0:
            writer.add_scalar('data/loss', total_loss/step_per_loss, i)
        print(score, i)

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
    '''
    env = "PongNoFrameskip-v4"
    agent = dqn.DQN(
        max_length=15000,
        state_size=[84, 84, 4],
        output_size=3,
        hidden=[512, 512],
        n_step=3,
        batch_size=32,
        gamma=0.999,
        lr=0.00025,
        train_size=1,
        activation=tf.nn.relu,
        update_size=300
    )
    '''
    #train(agent, env)
    test(agent, env)