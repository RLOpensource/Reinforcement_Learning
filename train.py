import tensorflow as tf
import gym
from wrappers import make_env
from tensorboardX import SummaryWriter
import dqn

def train(agent, env_name):
    writer = SummaryWriter()

    if env_name in ['MountainCar-v0', 'CartPole-v0']:
        sction = 1
        env = gym.make(env_name)
    elif env_name in ["PongNoFrameskip-v4", "BreakoutNoFrameskip-v4"]:
        sction = 2
        env = make_env(env_name)

    for i in range(99999):
        agent.epsilon = 1 / (i * 0.1 + 1)
        done = False
        state = env.reset()
        score = 0
        step_per_q_value = 0
        step_per_loss = 0
        sum_of_q_value = 0
        agent.memory.n_step.reset()
        while not done:
            if i % 10 == 0:
                env.render()
            action, q_value = agent.get_action(state, agent.epsilon)
            if not q_value == None:
                sum_of_q_value += q_value
                step_per_q_value += 1

            if sction == 1:
                next_state, reward, done, _ = env.step(action)
            elif sction == 2:
                next_state, reward, done, _ = env.step(action+1)

            score += reward
            agent.memory.append(state, next_state, reward, done, action)
            state = next_state
    
            if len(agent.memory.memory) > agent.batch_size:
                step_per_loss += 1
                loss = agent.update()
                agent.update_parameter()
        writer.add_scalar('data/score', score, i)
        writer.add_scalar('data/epsilon', agent.epsilon, i)
        writer.add_scalar('data/memory_size', len(agent.memory.memory), i)
        
        if not step_per_q_value == 0:
            writer.add_scalar('data/average_of_q_value', sum_of_q_value / step_per_q_value, i)
        if len(agent.memory.memory) > agent.batch_size:
            writer.add_scalar('data/loss', loss/step_per_loss, i)
        print(score, i)

if __name__ == '__main__':
    '''
    env = 'MountainCar-v0'
    agent = dqn.DQN(
        max_length=1e3,
        state_size=2,
        output_size=3,
        hidden=[256, 256, 256],
        n_step=10,
        batch_size=64,
        gamma=0.99,
        lr=0.001,
        train_size=1,
        activation=tf.nn.relu
    )
    '''
    env = "PongNoFrameskip-v4"
    agent = dqn.DQN(
        max_length=15000,
        state_size=[84, 84, 4],
        output_size=3,
        hidden=[512, 256],
        n_step=3,
        batch_size=32,
        gamma=0.999,
        lr=0.00025,
        train_size=1,
        activation=tf.nn.relu
    )
    

    train(agent, env)