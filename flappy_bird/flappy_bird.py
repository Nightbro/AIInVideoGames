import pygame
import random
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import csv

class Bird:
    def __init__(self, model_path=None):
        self.x = 60
        self.y = 300
        self.vel_y = 0
        self.width = 50
        self.height = 35
        self.gravity = 0.6
        self.flap = -10
        self.score = 0
        self.alive = True

        if model_path:
            self.model = load_model(model_path)
            #self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model = self.create_model()

        with open('flappy_bird_scores.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Bird Index', 'Score'])  # Header for the CSV

    def log_scores(self):
        # Log the scores for the current generation
        with open('flappy_bird_scores.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Generation', 'Bird Index', 'Score'])
            for idx, bird in enumerate(self.birds):
                writer.writerow([self.generation, idx, bird.score])

    def evolve(self):
        self.birds.sort(key=lambda b: b.score, reverse=True)
        
        # Log scores for the current generation
        self.log_scores()

        best_bird = self.birds[0]
        
        best_bird.model.save('best_flappy_bird_model.keras')
        
        next_generation = []
        for bird in self.birds[:3]:
            for _ in range(self.num_birds // 5):
                new_bird = Bird()
                new_bird.model.set_weights(bird.model.get_weights())
                self.mutate(new_bird)
                next_generation.append(new_bird)

        self.birds = next_generation[:self.num_birds]
        self.generation += 1
        for  bird in self.birds:
            print([self.generation,  bird.score])
        print(f"Generation: {self.generation}, Evolved!")

    def create_model(self):
        # Simple feedforward neural network
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Outputs a probability of flapping
        ])
        return model

    def decide(self, state):
        # Predict whether to flap or not
        state = np.array(state).reshape(1, -1)  # Reshape for the model input
        prediction = self.model.predict(state, verbose=0)[0, 0]
        return prediction > 0.5  # Flap if prediction > 0.5

    def jump(self):
        self.vel_y = self.flap

    def update(self):
        if self.alive:  # Only update score if the bird is still alive
            self.score += 1
            #print(self.score)
        self.vel_y += self.gravity
        self.y += self.vel_y

    def is_off_screen(self, height):
        return self.y > height - self.height or self.y < 0


class FlappyBirdGame:
    def __init__(self, num_birds=20):
        pygame.init()
        self.width, self.height = 400, 600
        self.win = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Flappy Bird NEAT")
        self.num_birds = num_birds

        # Game settings
        self.bird_width, self.bird_height = 50, 35
        self.pipe_width = 70
        self.pipe_gap = 200
        self.pipe_vel_x = -5
        self.pipe_min_height = 100
        self.pipe_max_height = 400

        self.pipes = [self.generate_pipe()]
        self.generation = 0

        self.background_img = pygame.image.load("background.png").convert()
        self.bird_img = pygame.image.load("bird1.png").convert_alpha()
        self.pipe_img = pygame.image.load("pipe.png").convert_alpha()

        self.background_img = pygame.transform.scale(self.background_img, (self.width, self.height))
        self.bird_img = pygame.transform.scale(self.bird_img, (self.bird_width, self.bird_height))
        self.pipe_img = pygame.transform.scale(self.pipe_img, (self.pipe_width, 400))

        self.birds = [Bird() for _ in range(self.num_birds)]

    def generate_pipe(self):
        height = random.randint(self.pipe_min_height, self.pipe_max_height)
        return {"x": self.width, "top": height - self.pipe_gap // 2, "bottom": height + self.pipe_gap // 2}

    def reset(self):
        self.pipes = [self.generate_pipe()]
        for bird in self.birds:
            bird.y = 300
            bird.vel_y = 0
            bird.score = 0
            bird.alive = True

    def get_state(self, bird):
        nearest_pipe = None
        for pipe in self.pipes:
            if pipe["x"] + self.pipe_width > bird.x:
                nearest_pipe = pipe
                break

        if not nearest_pipe:
            return None

        state = (
            bird.vel_y,
            nearest_pipe["x"] - bird.x,
            bird.y - nearest_pipe["top"],
            nearest_pipe["bottom"] - bird.y
        )
        return state

    def step(self):
        for bird in self.birds:
            if not bird.alive:
                continue
            state = self.get_state(bird)
            if state is not None:
                action = bird.decide(state)
                if action:
                    bird.jump()
            bird.update()
            bird.alive = not (bird.is_off_screen(self.height) or self.check_collision(bird))

        for pipe in self.pipes:
            pipe["x"] += self.pipe_vel_x

        if self.pipes[-1]["x"] < self.width - 300:
            self.pipes.append(self.generate_pipe())
        if self.pipes[0]["x"] < -self.pipe_width:
            self.pipes.pop(0)

    def check_collision(self, bird):
        for pipe in self.pipes:
            if (bird.x + bird.width > pipe["x"] and bird.x < pipe["x"] + self.pipe_width):
                if bird.y < pipe["top"] or bird.y + bird.height > pipe["bottom"]:
                    return True
        return False

    def evolve(self):
        self.birds.sort(key=lambda b: b.score, reverse=True)
        
        best_bird = self.birds[0]
        
        best_bird.model.save('best_flappy_bird_model.keras')
        #print(f"Model saved with score: {best_bird.score}")

        #print("Scores of the top birds in this generation:")
        #for bird in self.birds[:5]:
        #    print(bird.score)

        next_generation = []
        for bird in self.birds[:5]:
            for _ in range(self.num_birds // 5):
                new_bird = Bird()
                new_bird.model.set_weights(bird.model.get_weights())
                self.mutate(new_bird)
                next_generation.append(new_bird)

        self.birds = next_generation[:self.num_birds]
        self.generation += 1
        print(f"Generation: {self.generation}, Evolved!")

    def mutate(self, bird):
        weights = bird.model.get_weights()
        for i in range(len(weights)):
            if random.random() < 0.1:
                weights[i] += np.random.normal(0, 0.1, weights[i].shape)
        bird.model.set_weights(weights)

    def render(self):
        
        self.win.blit(self.background_img, (0, 0))
        for pipe in self.pipes:
            self.win.blit(self.pipe_img, (pipe["x"], pipe["top"] - self.pipe_img.get_height()))
            self.win.blit(self.pipe_img, (pipe["x"], pipe["bottom"]))
        for bird in self.birds:
            if bird.alive:
                self.win.blit(self.bird_img, (bird.x, bird.y))
        pygame.display.update()

    def is_done(self):
        return all(not bird.alive for bird in self.birds)

    def play_human(self):
        running = True
        clock = pygame.time.Clock()
        
        # Load the best model if it exists
        if os.path.exists('best_flappy_bird_model.keras'):
            print("Loading the best saved model...")
            for bird in self.birds:
                bird.model = tf.keras.models.load_model('best_flappy_bird_model.keras')

        while running:
            #clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        for bird in self.birds:
                            bird.jump()

            self.step()
            self.render()

            if self.is_done():
                self.evolve()
                self.reset() 
                print(bird.score for bird in self.birds)
                best_score = max(bird.score for bird in self.birds)
                print(f"Generation: {self.generation}, Best Score: {best_score}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False    

        pygame.quit()
    def play_human_s(self):
        running = True
        #clock = pygame.time.Clock()

        while running:
            #clock.tick(30)

            self.step()

            if self.is_done():
                self.evolve()
                self.reset()

                best_score = max(bird.score for bird in self.birds)
                for bird in self.birds:
                    print(bird.score)
                print(f"Generation: {self.generation}, Best Score: {best_score}")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()

    def run_trained_model(self):
        bird = Bird(model_path='best_flappy_bird_model.keras')

        running = True
        clock = pygame.time.Clock()

        while running:
            clock.tick(30)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get the current state and decide on an action
            state = self.get_state(bird)
            if state is not None:
                action = bird.decide(state)
                if action:
                    bird.jump()

            # Update bird and check for collisions
            bird.update()
            bird.alive = not (bird.is_off_screen(self.height) or self.check_collision(bird))

            # Update the game environment (pipes, ground, etc.)
            self.step()

            # Render the game
            self.render()

            # End the game if the bird dies
            if not bird.alive:
                print(f"Game Over! Score: {bird.score}")
                running = False

        pygame.quit()

if __name__ == "__main__":
    game = FlappyBirdGame(num_birds=5)
    
    #choice = input("Do you want to (p)lay the game, (t)rain or (r)un the trained model? ")

    #if choice.lower() == 't':
    game.play_human()
    #elif choice.lower() == 'r':
    #    game.run_trained_model()
    #elif choice.lower() == 'p':
    #    game.play_human()
