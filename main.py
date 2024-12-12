# Import libraries
import numpy as np

# Import libraries
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Global variables for datasets
x_train_mnist, y_train_mnist = None, None
x_train_cifar, y_train_cifar = None, None


def load_datasets():
    global x_train_mnist, y_train_mnist, x_train_cifar, y_train_cifar

    # MNIST
    (x_train_mnist, y_train_mnist), _ = tf.keras.datasets.mnist.load_data()
    print("MNIST dataset loaded.")  # Debugging log
    print(
        f"x_train_mnist shape: {x_train_mnist.shape}, y_train_mnist shape: {y_train_mnist.shape}"
    )
    x_train_mnist = x_train_mnist / 255.0
    x_train_mnist = x_train_mnist.reshape(-1, 28, 28, 1)  # Add channel dimension
    print(f"x_train_mnist reshaped to: {x_train_mnist.shape}")

    # CIFAR-10
    (x_train_cifar, y_train_cifar), _ = tf.keras.datasets.cifar10.load_data()
    print("CIFAR-10 dataset loaded.")  # Debugging log
    print(
        f"x_train_cifar shape: {x_train_cifar.shape}, y_train_cifar shape: {y_train_cifar.shape}"
    )
    x_train_cifar = x_train_cifar / 255.0


# Call the function to load datasets
load_datasets()


# Define Generator
def build_generator(latent_dim):
    """
    Generator creates a representation of a neural architecture.
    latent_dim: Dimension of the input noise vector.
    Outputs: [num_layers, num_neurons, activation_type]
    """
    print(f"Building generator with latent_dim: {latent_dim}")  # Debugging log
    model = Sequential(
        [
            Dense(64, activation="relu", input_dim=latent_dim),
            Dense(128, activation="relu"),
            Dense(3, activation="sigmoid"),  # Outputs 3 parameters
        ]
    )
    return model


# Instantiate the Generator
latent_dim = 10  # Dimension of noise input
generator = build_generator(latent_dim)
generator.summary()


# Define Discriminator
def build_discriminator():
    """
    Discriminator evaluates the plausibility of generated architectures.
    Input: [num_layers, num_neurons, activation_type]
    Output: Probability of being a plausible architecture.
    """
    print("Building discriminator.")  # Debugging log
    model = Sequential(
        [
            Dense(128, activation="relu", input_dim=3),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),  # Outputs probability
        ]
    )
    return model


# Instantiate the Discriminator
discriminator = build_discriminator()
print("Compiling discriminator.")  # Debugging log
discriminator.compile(optimizer="adam", loss="binary_crossentropy")
discriminator.summary()


# Combine Generator and Discriminator into a GAN
def build_gan(generator, discriminator):
    """
    Combines the generator and discriminator into a GAN.
    """
    print("Building GAN.")  # Debugging log
    discriminator.trainable = False  # Freeze Discriminator when training Generator
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer="adam", loss="binary_crossentropy")
    return gan


gan = build_gan(generator, discriminator)


# GAN Training Loop
def train_gan(generator, discriminator, gan, latent_dim, iterations=100):
    """
    Trains the GAN to generate plausible architectures.
    """
    print(f"Starting GAN training for {iterations} iterations.")  # Debugging log
    for iteration in range(iterations):
        # Step 1: Generate fake architectures
        noise = tf.random.normal((32, latent_dim))  # Batch size of 32
        generated_architectures = generator(noise)
        print(
            f"Iteration {iteration}: Generated architectures shape: {generated_architectures.shape}"
        )  # Debugging log

        # Step 2: Create real and fake labels
        real_labels = tf.ones((32, 1))
        fake_labels = tf.zeros((32, 1))

        # Step 3: Train Discriminator
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(generated_architectures, fake_labels)
        print(f"Iteration {iteration}: Discriminator loss: {d_loss}")  # Debugging log

        # Step 4: Train Generator via the GAN
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, real_labels)
        print(f"Iteration {iteration}: Generator loss: {g_loss}")  # Debugging log

        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(
                f"Iteration {iteration}/{iterations}: Discriminator Loss: {discriminator.evaluate(generated_architectures, fake_labels, verbose=0)}"
            )


# Training the GAN Testing the GAN
train_gan(generator, discriminator, gan, latent_dim, iterations=50)

### Testing the GAN

# Generate a noise vector
noise = tf.random.normal((1, latent_dim))
print(f"Generated noise vector for testing: {noise.numpy()}")  # Debugging log

# Generate a sample architecture
sample_architecture = generator(noise)
print(f"Sample architecture generated: {sample_architecture.numpy()}")  # Debugging log

try:
    # Force conversion to a NumPy array
    sample_architecture_np = np.array(sample_architecture.numpy().squeeze())
    print(
        f"Converted sample architecture to NumPy array: {sample_architecture_np}"
    )  # Debugging log

    # Rescale the output if values are too small or unusual
    sample_architecture_rescaled = np.clip(sample_architecture_np * 1000, 0, 1)
    print(
        f"Rescaled sample architecture: {sample_architecture_rescaled}"
    )  # Debugging log

    # Check for valid values in rescaled output
    if not np.all(np.isfinite(sample_architecture_rescaled)):
        print("Error: Rescaled architecture contains invalid values (NaN or Inf).")
    else:
        # Print the valid architecture
        print("Sample Generated Architecture (Rescaled):", sample_architecture_rescaled)
except Exception as e:
    print("Error during conversion or validation:", e)

# Debugging information for raw tensor
try:
    print("Raw Architecture Output (Tensor):", sample_architecture)
except Exception as e:
    print("Error printing raw tensor:", e)

# Debugging Tensor type and shape
print("Output Type:", type(sample_architecture))
print("Output Shape:", sample_architecture.shape)


def decode_architecture(output):
    """
    Decodes the rescaled architecture into meaningful values.
    output: Array of 3 values [num_layers, num_neurons, activation_type].
    Returns: Decoded architecture components.
    """
    print(f"Decoding architecture output: {output}")  # Debugging log

    # Ensure the output is a valid NumPy array
    if isinstance(output, tf.Tensor):
        output = output.numpy().squeeze()
    if isinstance(output, np.ndarray):
        output = np.squeeze(output)

    # Decode the architecture
    num_layers = int(output[0] * 4) + 1  # Map to 1–5 layers
    num_neurons = int(output[1] * 224) + 32  # Map to 32–256 neurons
    activation = "relu" if output[2] < 0.5 else "tanh"  # Choose activation function
    decoded = (num_layers, num_neurons, activation)
    print(f"Decoded architecture: {decoded}")  # Debugging log
    return decoded


# Decode the generated architecture
try:
    decoded_architecture = decode_architecture(sample_architecture_rescaled)
    print("Decoded Architecture:", decoded_architecture)
except Exception as e:
    print("Error decoding architecture:", e)


# ------------------ Reward Function Block -------------------
def evaluate_architecture(architecture, x_train, y_train):
    """
    Evaluates a generated architecture by training it on a subset of the data.
    architecture: Tuple (num_layers, num_neurons, activation)
    x_train, y_train: Dataset for evaluation.
    Returns: Reward (validation accuracy).
    """
    num_layers, num_neurons, activation = architecture

    print(
        f"Evaluating architecture: Layers={num_layers}, Neurons={num_neurons}, Activation={activation}"
    )  # Debugging log

    # Build the model
    model = Sequential()
    model.add(Dense(num_neurons, activation=activation, input_shape=(28 * 28,)))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))

    model.add(Dense(10, activation="softmax"))  # Output layer for classification
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Train the model on a small subset of data
    history = model.fit(
        x_train[:1000].reshape(-1, 28 * 28),  # Flatten images for Dense layers
        y_train[:1000],
        epochs=1,
        batch_size=32,
        validation_split=0.1,
        verbose=0,
    )

    # Get the validation accuracy as the reward
    reward = history.history["val_accuracy"][-1]
    print(f"Validation accuracy for architecture: {reward}")  # Debugging log

    return reward


# Example usage
reward = evaluate_architecture(decoded_architecture, x_train_mnist, y_train_mnist)
print("Reward for Generated Architecture:", reward)

# ------------------ RL Agent Block -------------------
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gym import spaces
from stable_baselines3 import PPO
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class ArchitectureEnv(gym.Env):
    """
    Custom Environment for RL agent to evaluate neural architectures.
    """

    def __init__(self, x_train, y_train):
        super(ArchitectureEnv, self).__init__()
        self.x_train = x_train
        self.y_train = y_train

        # Define action and observation space
        # Actions: [num_layers, num_neurons, activation_type]
        self.action_space = spaces.Box(
            low=np.array([1, 32, 0]), high=np.array([5, 256, 1]), dtype=np.float32
        )

        # Observation space: None, as we only care about actions
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def reset(self):
        """Reset environment state."""
        self.state = np.zeros(1)
        return self.state

    def step(self, action):
        """Perform an action and return the reward."""
        num_layers = int(action[0])
        num_neurons = int(action[1])
        activation = "relu" if action[2] < 0.5 else "tanh"

        architecture = (num_layers, num_neurons, activation)
        reward = evaluate_architecture(architecture, self.x_train, self.y_train)

        done = True  # Each step completes an episode
        return self.state, reward, done, {}


# Function to generate architecture given a dataset
def generate_architecture(generator, latent_dim):
    """
    Generates a neural network architecture based on the GAN generator output.
    generator: Trained GAN generator.
    latent_dim: Dimension of the latent input for the generator.
    Returns: Decoded architecture tuple (num_layers, num_neurons, activation).
    """
    noise = tf.random.normal((1, latent_dim))
    generated_architecture = generator(noise).numpy().squeeze()
    decoded_architecture = decode_architecture(generated_architecture)
    return decoded_architecture


# Visualization Function
def visualize_architecture(architecture):
    """
    Visualizes the generated architecture as a block diagram.
    architecture: Tuple (num_layers, num_neurons, activation).
    """
    num_layers, num_neurons, activation = architecture
    plt.figure(figsize=(8, 4))
    for i in range(num_layers):
        plt.bar(i, num_neurons, color="blue", alpha=0.6)
        plt.text(i, num_neurons + 5, f"Layer {i + 1}\n{activation}", ha="center")
    plt.title("Generated Neural Network Architecture")
    plt.xlabel("Layers")
    plt.ylabel("Neurons")
    plt.ylim(0, num_neurons + 50)
    plt.show()


# ------------------ GAN Training -------------------
print("Training GAN.")


def train_gan(generator, discriminator, gan, latent_dim, iterations=50):
    for iteration in range(iterations):
        # Step 1: Generate fake architectures
        noise = tf.random.normal((32, latent_dim))  # Batch size of 32
        generated_architectures = generator(noise)

        # Step 2: Create labels for training discriminator
        real_labels = tf.ones((32, 1))
        fake_labels = tf.zeros((32, 1))

        # Step 3: Train Discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(generated_architectures, fake_labels)

        # Step 4: Train Generator via GAN
        discriminator.trainable = False
        gan.train_on_batch(noise, real_labels)


# ------------------ RL Agent Training -------------------
print("Training RL agent.")
# Create environment
(x_train_mnist, y_train_mnist), _ = tf.keras.datasets.mnist.load_data()
x_train_mnist = x_train_mnist / 255.0
x_train_mnist = x_train_mnist.reshape(-1, 28, 28)
y_train_mnist = y_train_mnist

env = ArchitectureEnv(x_train_mnist, y_train_mnist)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    n_steps=32,  # Reduce number of steps per update
    batch_size=8,  # Smaller training batches
    n_epochs=1,  # Fewer optimization epochs
)
model.learn(total_timesteps=1)  # Reduced timesteps for quick testing


# ------------------ Generate and Return Architecture -------------------
def generate_neural_network(dataset_name):
    """
    Input: Dataset name (e.g., "MNIST" or "CIFAR-10").
    Output: A decoded neural network architecture.
    """
    if dataset_name == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        input_shape = (28, 28)
    elif dataset_name == "CIFAR-10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        input_shape = (32, 32, 3)
    else:
        raise ValueError("Unsupported dataset. Please use 'MNIST' or 'CIFAR-10'.")

    # Generate architecture using the trained GAN generator
    architecture = generate_architecture(generator, latent_dim)
    print(f"Generated Architecture for {dataset_name}: {architecture}")

    # Visualize architecture
    visualize_architecture(architecture)

    # Train and evaluate the generated architecture
    def train_and_evaluate_model(
        architecture, x_train, y_train, x_test, y_test, input_shape
    ):
        """
        Trains and evaluates a generated architecture on the given dataset.

        Parameters:
            architecture: Tuple (num_layers, num_neurons, activation).
            x_train, y_train: Training data and labels.
            x_test, y_test: Testing data and labels.
            input_shape: Shape of the input data (e.g., (28, 28, 1) for MNIST).
        """
        num_layers, num_neurons, activation = architecture

        # Build model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        for _ in range(num_layers):
            model.add(tf.keras.layers.Dense(num_neurons, activation=activation))
        model.add(
            tf.keras.layers.Dense(10, activation="softmax")
        )  # 10 classes for both MNIST and CIFAR

        # Compile model
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train model
        model.fit(
            x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
        )

        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
        print(f"Test Accuracy: {test_accuracy}")
        return test_accuracy

    test_accuracy = train_and_evaluate_model(
        architecture, x_train, y_train, x_test, y_test, input_shape
    )
    print(f"{dataset_name} Test Accuracy: {test_accuracy}")
    return architecture, test_accuracy


# Example usage
generated_architecture_mnist, test_accuracy_mnist = generate_neural_network("MNIST")
generated_architecture_cifar, test_accuracy_cifar = generate_neural_network("CIFAR-10")

# ------------------ Documentation -------------------
"""
Ensure to include comprehensive documentation explaining:
1. The GAN-RL system and how it generates, evaluates, and refines architectures.
2. Training details for the GAN and RL components.
3. Example outputs and their corresponding performance metrics.
4. Scripts required to reproduce the results.
"""
