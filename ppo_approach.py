import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import random
from collections import Counter
from tqdm import tqdm
import os
import math

RL_TRAIN_DICT_PATH = "/kaggle/input/hman-ds/rl_train.txt"
FINAL_TEST_DICT_PATH = "/kaggle/input/hman-ds/final_test.txt"

# key model we are using for transfer learning (the 62% supervised model)
PRETRAINED_62_MODEL_PATH = "/kaggle/input/hman-model-62/bilstm_attn_hangman_gated_62.pth"
MODEL_FILE = "ppo_end_to_end_finetuned_from_62.pth"

# seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# hyperparams
D_MODEL = 128
HIDDEN_DIM = 256
NUM_LAYERS = 4
ATTN_HEADS = 8
MAX_SEQ_LEN = 32
ALPHABET_LEN = 26
STATE_DIM_HISTORY = ALPHABET_LEN * 2

CHAR_TO_IX = {char: i + 2 for i, char in enumerate("abcdefghijklmnopqrstuvwxyz")}
CHAR_TO_IX['_'] = 1
VOCAB_SIZE = len(CHAR_TO_IX) + 1
# maps action index -> (0-25) to a letter
ACTION_TO_CHAR = {i: char for i, char in enumerate("abcdefghijklmnopqrstuvwxyz")}

# PPO params
LEARNING_RATE = 3e-5
PPO_EPOCHS = 10
CLIP_EPSILON = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
BATCH_SIZE = 2048
NUM_EPISODES = 20000

class ActorCriticHangmanModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, state_dim_history, num_heads):
        super(ActorCriticHangmanModel, self).__init__()

        self.lstm_output_dim = hidden_dim * 2

        # The feat extraction backbone from the original model
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, batch_first=True
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.lstm_output_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(self.lstm_output_dim)
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim_history, self.lstm_output_dim),
            nn.ReLU(),
            nn.Linear(self.lstm_output_dim, self.lstm_output_dim)
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(self.lstm_output_dim * 2, self.lstm_output_dim),
            nn.ReLU(),
            nn.Linear(self.lstm_output_dim, self.lstm_output_dim),
            nn.Sigmoid()
        )

        # New PPO Actor and Critic heads replace the old classification layer
        self.actor_head = nn.Linear(self.lstm_output_dim, ALPHABET_LEN)
        self.critic_head = nn.Linear(self.lstm_output_dim, 1)

    def _get_shared_features(self, x, state_52d):
        attn_key_padding_mask = (x == 0)
        embedded = self.embedding(x)
        seq_len = x.size(1)
        lstm_out, _ = self.lstm(embedded)
        attn_output, _ = self.attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=attn_key_padding_mask
        )
        O_Attn = self.norm(lstm_out + attn_output)

        C_State_projected = self.state_projection(state_52d)
        C_State = C_State_projected.unsqueeze(1).repeat(1, seq_len, 1)

        combined_features = torch.cat([O_Attn, C_State], dim=-1)
        Gamma = self.gate_fc(combined_features)

        O_Gated = Gamma * O_Attn + (1 - Gamma) * C_State

        batch_size = x.size(0)
        first_blank_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        # the position of the first blank '_' to extract the sequence feature
        for i in range(batch_size):
            blanks = (x[i] == 1).nonzero(as_tuple=True)[0]
            first_blank_indices[i] = blanks[0] if len(blanks) > 0 else 0

        idx_expanded = first_blank_indices.view(-1, 1, 1).expand(-1, 1, self.lstm_output_dim)
        shared_features = O_Gated.gather(1, idx_expanded).squeeze(1)

        return shared_features

    def forward(self, x, state_52d):
        shared_features = self._get_shared_features(x, state_52d)
        action_logits = self.actor_head(shared_features)
        state_value = self.critic_head(shared_features)
        return action_logits, state_value

    def load_pretrained_weights(self, old_model_path):
        if not os.path.exists(old_model_path):
            print(f"WARNING: Pre-trained model not found at {old_model_path}. Starting from scratch.")
            return

        print(f"Loading pre-trained weights from {old_model_path}...")
        old_state_dict = torch.load(old_model_path, map_location=torch.device('cpu'))
        new_state_dict = self.state_dict()

        transplanted_count = 0
        skipped_count = 0
        # copy weights from the old model to the new model's shared layers
        for name, param in new_state_dict.items():
            if name in old_state_dict:
                if old_state_dict[name].shape == param.shape:
                    param.copy_(old_state_dict[name])
                    transplanted_count += 1
                else:
                    skipped_count += 1
            else:
                # Expected to skip actor/critic heads for random initialization
                skipped_count += 1
        
        print(f"Successfully transplanted {transplanted_count} layers.")


class PPOAgent:
    def __init__(self, model_path=MODEL_FILE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent using device: {self.device}")

        self.network = ActorCriticHangmanModel(
            VOCAB_SIZE, D_MODEL, HIDDEN_DIM, NUM_LAYERS,
            STATE_DIM_HISTORY, ATTN_HEADS
        ).to(self.device)

        self.model_path = model_path
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE, eps=1e-5)
        self.memory = []

    def load_model(self, pretrained_path=PRETRAINED_62_MODEL_PATH):
        if os.path.exists(self.model_path):
            print(f"Loading fine-tuned PPO Agent from {self.model_path}...")
            self.network.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print("No fine-tuned agent found. Performing transfer learning from 62% model...")
            self.network.load_pretrained_weights(pretrained_path)

    def save_model(self):
        print(f"Saving PPO agent model to {self.model_path}")
        torch.save(self.network.state_dict(), self.model_path)

    def clear_memory(self):
        self.memory = []

    def store_transition(self, state_tuple, action, log_prob, reward, done):
        self.memory.append((state_tuple, action, log_prob, reward, done))

    def get_state_tensors(self, pattern, guessed_letters_set):
        # prep inpt seq (pattern indices)
        input_seq_list = []
        for c in pattern:
            input_seq_list.append(CHAR_TO_IX.get(c, 0))
        input_seq = torch.tensor([input_seq_list], dtype=torch.long).to(self.device)

        # prep 52D history state
        visible_letters = set(c for c in pattern if c.isalpha())
        incorrect_guesses = guessed_letters_set - visible_letters

        state_vector = torch.zeros(STATE_DIM_HISTORY)
        for char in visible_letters:
            char_idx = CHAR_TO_IX.get(char, 0) - 2
            if 0 <= char_idx < ALPHABET_LEN: state_vector[char_idx] = 1.0
        for char in incorrect_guesses:
            char_idx = CHAR_TO_IX.get(char, 0) - 2
            if 0 <= char_idx < ALPHABET_LEN: state_vector[char_idx + ALPHABET_LEN] = 1.0

        state_52d = state_vector.unsqueeze(0).to(self.device)
        return input_seq, state_52d

    def select_action(self, pattern, guessed_letters_set):
        self.network.eval()
        input_seq, state_52d = self.get_state_tensors(pattern, guessed_letters_set)

        with torch.no_grad():
            action_logits, _ = self.network(input_seq, state_52d)

        # masking already guessed letters by setting their logits to -inf
        mask = torch.full_like(action_logits, -float('inf'))
        valid_actions = [i for i in range(ALPHABET_LEN) if ACTION_TO_CHAR[i] not in guessed_letters_set]

        if not valid_actions: return None, None

        mask[:, valid_actions] = 0.0
        masked_logits = action_logits + mask

        # sample action based on the prob distribution
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.network.train()
        return action.item(), log_prob.item()

    def update(self):
        self.network.train()

        state_tuples, actions, log_probs_old, rewards, dones = zip(*self.memory)

        # process and pad sequences for batch processing
        input_seq_list = [s[0].squeeze(0) for s in state_tuples]
        input_seqs = torch.nn.utils.rnn.pad_sequence(
            input_seq_list, batch_first=True, padding_value=0
        )
        if input_seqs.size(1) < MAX_SEQ_LEN:
             pad_size = MAX_SEQ_LEN - input_seqs.size(1)
             input_seqs = F.pad(input_seqs, (0, pad_size), "constant", 0)
        state_52ds = torch.cat([s[1] for s in state_tuples])

        # prep tensors, excluding the dummy terminal state added for GAE
        actions = torch.tensor(actions[:-1], dtype=torch.long, device=self.device)
        log_probs_old = torch.tensor(log_probs_old[:-1], dtype=torch.float32, device=self.device)

        # calc V(s) for GAE
        with torch.no_grad():
            _, values = self.network(input_seqs, state_52ds)
            values = values.squeeze()

        # Generalized Advantage Estimation (GAE) calc
        advantages = []
        gae = 0.0
        for t in reversed(range(BATCH_SIZE)):
            is_done_mask = 1.0 - torch.tensor(dones[t], dtype=torch.float32, device=self.device)
            delta = rewards[t] + GAMMA * values[t+1] * is_done_mask - values[t]
            gae = delta + GAMMA * GAE_LAMBDA * is_done_mask * gae
            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        input_seqs_train = input_seqs[:-1]
        state_52ds_train = state_52ds[:-1]

        # PPO optimization loop
        for _ in range(PPO_EPOCHS):
            logits_new, values_new = self.network(input_seqs_train, state_52ds_train)
            values_new = values_new.squeeze()

            dist = Categorical(logits=logits_new)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO clipped objective for actor loss
            ratio = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic Loss (val fn loss)
            critic_loss = F.mse_loss(values_new, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        self.clear_memory()
        return loss.item()


class LocalGameSimulator:
    def __init__(self, secret_word, max_guesses=6):
        self.secret_word = secret_word.lower()
        self.max_guesses = max_guesses
        self.lives_remaining = self.max_guesses
        self.guessed_letters = set()
        self.pattern = ['_'] * len(self.secret_word)

    def get_pattern(self): return "".join(self.pattern)
    def is_won(self): return '_' not in self.pattern
    def is_lost(self): return self.lives_remaining <= 0
    def is_game_over(self): return self.is_won() or self.is_lost()

    def guess(self, letter_char):
        if not isinstance(letter_char, str) or len(letter_char) != 1 or not 'a' <= letter_char <= 'z':
            return "invalid", -1.0

        if letter_char in self.guessed_letters:
            return "repeat", -1.0

        self.guessed_letters.add(letter_char)

        if letter_char in self.secret_word:
            for i, char in enumerate(self.secret_word):
                if char == letter_char:
                    self.pattern[i] = letter_char
            
            # reward structure is critical for RL performance
            if self.is_won():
                return "win", 10.0
            else:
                return "correct", 1.0
        else:
            self.lives_remaining -= 1
            if self.is_lost():
                return "loss", -10.0
            else:
                return "incorrect", -2.0


def run_rl_training(rl_train_words):
    print("\n Training the PPO Agent ")
    agent = PPOAgent()
    agent.load_model(pretrained_path=PRETRAINED_62_MODEL_PATH)

    current_step = 0
    running_reward = 0
    avg_reward_window = 100
    loss_str = ""

    with tqdm(range(1, NUM_EPISODES + 1), desc="Training PPO") as pbar:
        for i_episode in pbar:
            game = LocalGameSimulator(random.choice(rl_train_words))
            episode_reward = 0

            while not game.is_game_over():
                state_tuple = agent.get_state_tensors(game.get_pattern(), game.guessed_letters)
                action, log_prob = agent.select_action(game.get_pattern(), game.guessed_letters)

                if action is None:
                    tqdm.write(f"Warning: No valid actions in episode {i_episode}. Breaking.")
                    break

                letter_to_guess = ACTION_TO_CHAR[action]
                result, reward = game.guess(letter_to_guess)
                done = game.is_game_over()

                agent.store_transition(state_tuple, action, log_prob, reward, done)
                current_step += 1
                episode_reward += reward

                # checks if we have enough steps for a PPO update rollout
                if current_step >= BATCH_SIZE:
                    # stores last state for accur GAE calc
                    last_state_tuple = agent.get_state_tensors(game.get_pattern(), game.guessed_letters)
                    agent.store_transition(last_state_tuple, 0, 0.0, 0.0, True)

                    loss = agent.update()
                    loss_str = f", Loss={loss:.4f}"
                    current_step = 0

                pbar.set_postfix_str(f"Steps={current_step}/{BATCH_SIZE}{loss_str}")

                if done:
                    break

            running_reward += episode_reward

            if i_episode % avg_reward_window == 0:
                avg_reward = running_reward / avg_reward_window
                tqdm.write(f"Episode {i_episode}\tAvg Reward: {avg_reward:.2f}")
                running_reward = 0

            if i_episode % 1000 == 0:
                tqdm.write(f"\nEpisode {i_episode}. Saving model...")
                agent.save_model()

    agent.save_model()
    return agent

def run_final_evaluation(agent, test_words):
    print("\n Evaluating Final PPO Agent ")
    wins = 0
    agent.network.eval()

    for secret_word in tqdm(test_words, desc="Final Evaluation"):
        game = LocalGameSimulator(secret_word)

        while not game.is_game_over():
            input_seq, state_52d = agent.get_state_tensors(game.get_pattern(), game.guessed_letters)

            with torch.no_grad():
                action_logits, _ = agent.network(input_seq, state_52d)

            # masking for deterministic inference, argmax
            mask = torch.full_like(action_logits, -float('inf'))
            valid_actions = [i for i in range(ALPHABET_LEN) if ACTION_TO_CHAR[i] not in game.guessed_letters]

            if not valid_actions:
                break

            mask[:, valid_actions] = 0.0
            masked_logits = action_logits + mask

            action = torch.argmax(masked_logits).item()
            letter_to_guess = ACTION_TO_CHAR[action]

            game.guess(letter_to_guess)

        if game.is_won():
            wins += 1

    accuracy = (wins / len(test_words)) * 100
    print(f"\n FINAL ACCURACY: {accuracy:.2f}% ")
    return accuracy

if __name__ == '__main__':
    print(" Loading Data Splits ")
    try:
        # filter words
        with open(RL_TRAIN_DICT_PATH, "r") as f:
             rl_train_words = [line.strip().lower() for line in f if line.strip() and len(line.strip()) < MAX_SEQ_LEN]
        with open(FINAL_TEST_DICT_PATH, "r") as f:
             final_test_words = [line.strip().lower() for line in f if line.strip() and len(line.strip()) < MAX_SEQ_LEN]
        print(f"Loaded {len(rl_train_words)} words for RL training.")
        print(f"Loaded {len(final_test_words)} words for final testing.")
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found. {e}"); exit()

    if not os.path.exists(PRETRAINED_62_MODEL_PATH):
        print(f"\nERROR: Pre-trained 62% model not found at '{PRETRAINED_62_MODEL_PATH}'.")
        print("This script requires it for transfer learning."); exit()

    print(f"\nStarting PPO fine-tuning for {NUM_EPISODES} episodes...")
    agent = run_rl_training(rl_train_words)

    print("\nStarting final evaluation...")
    run_final_evaluation(agent, final_test_words)

    print("\nScript finished.")