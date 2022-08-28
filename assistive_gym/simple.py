import gym
import numpy as np
import assistive_gym
from numpngw import write_apng
from matplotlib import pyplot as plt

# Make a feeding assistance environment with the Jaco robot.
env = gym.make('MoveToDot-v0')
env.set_seed(200)
# Setup a camera in the environment to capture images (for rendering)
env.setup_camera(camera_eye=[-0.5, -0.75, 1.5], camera_target=[-0.2, 0, 0.75], fov=80, camera_width=1920//4, camera_height=1080//4)

# Reset the environment
observation = env.reset()
imG, depth = env.get_camera_image_depth()
frames = []
done = False
for i in range(1): 
#while not done:
    # Step the simulation forward. Have the robot take a random action.
    observation, reward, done, info = env.step(env.action_space.sample())
    # Capture (render) an image from the camera
    mask = (observation['mask'] == 2)
    img, depth = env.get_camera_image_depth()
    frames.append(img)
    plt.imshow(mask, interpolation='nearest')
    plt.show()
env.disconnect()

# Compile the camera frames into an animated png (apng)
write_apng('output.png', frames, delay=100)

plt.imshow(imG, interpolation='nearest')
plt.show()
