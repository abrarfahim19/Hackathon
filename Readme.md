# Neural Architecture Design with GAN and RL

This repository contains a system that combines Generative Adversarial Networks (GANs) and Reinforcement Learning (RL) to automatically design, optimize, and evaluate neural network architectures.

## Overview
The system automates the process of neural architecture search by integrating GANs and RL. GANs propose candidate architectures, while RL refines these architectures iteratively. The generated architectures are then visualized and tested on datasets like MNIST and CIFAR-10.

---

## Key Features

1. **Dataset Loading**
   - Loads and preprocesses datasets (e.g., MNIST and CIFAR-10).
   - Normalizes data and reshapes it for training.

2. **GAN-Based Architecture Generation**
   - **Generator**: Proposes neural network architectures by generating parameters like the number of layers, neurons, and activation functions.
   - **Discriminator**: Evaluates the plausibility of the generated architectures.

3. **Reinforcement Learning (RL)**
   - Custom RL environment to simulate the evaluation of neural architectures.
   - Uses PPO (Proximal Policy Optimization) as the RL agent to refine architectures based on rewards.

4. **Feedback Loop**
   - Integrates GAN and RL in a feedback loop for iterative improvement of generated architectures.

5. **Architecture Visualization**
   - Generates bar charts to visualize layers, neurons, and activation functions of the designed architectures.

6. **Model Training and Evaluation**
   - Trains and tests generated architectures on MNIST and CIFAR-10 datasets.
   - Reports accuracy for each architecture.

---

## Code Structure

### Dataset Loading
```python
# Load datasets and preprocess
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
x_train_mnist = x_train_mnist / 255.0
x_test_mnist = x_test_mnist / 255.0
```

### GAN Components
- **Generator**: Proposes neural network architectures.
```python
def build_generator(latent_dim):
    model = Sequential([
        Dense(64, activation="relu", input_dim=latent_dim),
        Dense(128, activation="relu"),
        Dense(3, activation="sigmoid")  # Outputs [layers, neurons, activation]
    ])
    return model
```
- **Discriminator**: Evaluates the plausibility of generated architectures.
```python
def build_discriminator():
    model = Sequential([
        Dense(128, activation="relu", input_dim=3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")  # Outputs probability
    ])
    return model
```

### RL Components
- **Custom Environment**: Evaluates architectures and provides rewards.
```python
class ArchitectureEnv(gym.Env):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.action_space = spaces.Box(low=np.array([1, 32, 0]), high=np.array([5, 256, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        architecture = decode_action(action)
        reward = evaluate_architecture(architecture, self.x_train, self.y_train)
        return self.state, reward, True, {}
```
- **RL Agent**: Uses PPO for refining architectures.
```python
model = PPO("MlpPolicy", env, verbose=1, n_steps=32, batch_size=8, n_epochs=1)
model.learn(total_timesteps=100)
```

### Feedback Loop
Integrates GAN and RL for iterative refinement.
```python
def feedback_loop(generator, rl_agent, latent_dim, iterations=2):
    for iteration in range(iterations):
        noise = tf.random.normal((3, latent_dim))
        generated_architectures = generator(noise).numpy()
        for arch in generated_architectures:
            _, reward, _, _ = env.step(arch)
```

### Visualization
```python
def visualize_architecture(architecture):
    num_layers, num_neurons, activation = architecture
    plt.bar(range(num_layers), [num_neurons] * num_layers)
    plt.title("Generated Neural Network Architecture")
    plt.show()
```

### Testing Generated Architectures
```python
def train_and_evaluate_model(architecture, x_train, y_train, x_test, y_test, input_shape):
    model = Sequential([Flatten(input_shape=input_shape),
                        *[Dense(num_neurons, activation=activation) for _ in range(num_layers)],
                        Dense(10, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, validation_split=0.1)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    return test_accuracy
```

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abrarfahim19/Hackathon.git
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow gym matplotlib stable-baselines3
   ```

---

## Usage
1. Train the GAN:
   ```python
   train_gan(generator, discriminator, gan, latent_dim=10, iterations=50)
   ```
2. Train RL agent:
   ```python
   model.learn(total_timesteps=100)
   ```
3. Generate and test architecture:
   ```python
   generated_architecture, test_accuracy = generate_neural_network("MNIST")
   ```

---

## Results
- Example generated architecture for MNIST: `(1 layer, 33 neurons, relu)`
- Test accuracy on MNIST: `~0.85`

---

## Future Work
1. Support for additional datasets (e.g., ImageNet).
2. Extend RL agent to explore more complex architectures.
3. Optimize GAN training for faster convergence.

---

## Contributing
Feel free to submit issues or pull requests to improve this project.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

