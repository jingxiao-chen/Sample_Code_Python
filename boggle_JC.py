########################################
##########                    ##########
########## PH 1975 Homework 2 ##########
##########                    ##########
########################################
## Jingxiao Chen, Fangyu Li, Sepideh Saroukhani
##########   Sep. 29, 2019    ##########
########################################
##########      Boggle        ##########
########################################

import enchant
import random
import numpy

# random.seed(2)

##########  Construct a board  ##########
# 16 dice
die1 = ["A", "E", "A", "N", "E", "G"]
die2 = ["A", "H", "S", "P", "C", "O"]
die3 = ["A", "S", "P", "F", "F", "K"]
die4 = ["O", "B", "J", "O", "A", "B"]
die5 = ["I", "O", "T", "M", "U", "C"]
die6 = ["R", "Y", "V", "D", "E", "L"]
die7 = ["L", "R", "E", "I", "X", "D"]
die8 = ["E", "I", "U", "N", "E", "S"]
die9 = ["W", "N", "G", "E", "E", "H"]
die10 = ["L", "N", "H", "N", "R", "Z"]
die11 = ["T", "S", "T", "I", "Y", "D"]
die12 = ["O", "W", "T", "O", "A", "T"]
die13 = ["E", "R", "T", "T", "Y", "L"]
die14 = ["T", "O", "E", "S", "S", "I"]
die15 = ["T", "E", "R", "W", "H", "V"]
die16 = ["N", "U", "I", "H", "M", "Qu"]

# random select one side of each die and construct a 4x4 grid
die1_rand = random.choice(die1)
die2_rand = random.choice(die2)
die3_rand = random.choice(die3)
die4_rand = random.choice(die4)
die5_rand = random.choice(die5)
die6_rand = random.choice(die6)
die7_rand = random.choice(die7)
die8_rand = random.choice(die8)
die9_rand = random.choice(die9)
die10_rand = random.choice(die10)
die11_rand = random.choice(die11)
die12_rand = random.choice(die12)
die13_rand = random.choice(die13)
die14_rand = random.choice(die14)
die15_rand = random.choice(die15)
die16_rand = random.choice(die16)

dice_rand = [die1_rand, die2_rand, die3_rand, die4_rand,
             die5_rand, die6_rand, die7_rand, die8_rand,
             die9_rand, die10_rand, die11_rand, die12_rand,
             die13_rand, die14_rand, die15_rand, die16_rand]
dice_shuffled = random.sample(dice_rand, len(dice_rand))
grid = numpy.array(dice_shuffled).reshape(4, 4)


# Check the word if should be scored followed by following rules
# 1. The word is not scored
def is_scored(word):
    if word.upper() not in scored_words:
        return True
    else:
        print("The word", word, "has already been used.")
        return False


# 2. at least three letters long
def is_long_enough(word):
    if len(word) >= 3:
        return True
    else:
        print("The word", word, "is too short.")
        return False


# 3. Lookup the word if in the dictionary
def dictionary_check(word):
    en_dict = enchant.Dict("en_US")
    if en_dict.check(word):
        return True
    else:
        print("The word", word, "is ... not a word.")
        return False


# 4. The word is present in the grid
## Master function: returns true if the word is in the grid
def word_in_grid(board, word):
    # Convert grid to a dict object
    word_upper = word.upper()
    grid_dict = dict(((i, j), board[i][j]) for i in range(len(board)) for j in range(len(board[0])))
    used_pos = []
    if any(is_word_present(grid_dict, key, word_upper, used_pos) for (key, val) in dict.items(grid_dict)):
        return True
    else:
        print("The word", word, "is not present.")
        return False


## Check if the word input in the grid dictionary
def is_word_present(board, location, word, used_pos):
    if word == "":
        return True
    pos_paths = letter_index(board, location[0], location[1], word[0])
    used_pos.append(pos_paths)
    if pos_paths in used_pos[:-1]:
        return False
    if not pos_paths:
        return False
    return any(is_word_present(board, the_pos, word[1:], used_pos) for the_pos in pos_paths)


## Returns a list of positions of the letter
def letter_index(board, r, c, letter):
    positions = [(r - 1, c - 1), (r - 1, c), (r - 1, c + 1),
                 (r, c - 1), (r, c + 1), (r + 1, c - 1),
                 (r + 1, c), (r + 1, c + 1)]
    return [p for p in positions if p in board and board[p] == letter]

# 5. Calculate scores
def calculate_score(word):
    if len(word) == 3 or len(word) == 4:
        print("The word", word, "is worth 1 point.")
        return 1
    if len(word) == 5:
        print("The word", word, "is worth 2 point.")
        return 2
    if len(word) == 6:
        print("The word", word, "is worth 3 point.")
        return 3
    if len(word) == 7:
        print("The word", word, "is worth 5 point.")
        return 5
    if len(word) > 8:
        print("The word", word, "is worth 11 point.")
        return 11


scored_words = []

# Main function #
def boggle(board, word_list):
    score_tmp = 0
    for word in word_list:
        if is_scored(word):
            # print(word, "is not scored")
            if is_long_enough(word):
                # print(word, "is long enough")
                if dictionary_check(word):
                    # print(word, "is a word")
                    if word_in_grid(board, word):
                        # print(word, "is in the grid.")
                        scored_words.append(word.upper())
                        s = calculate_score(word)
                        score_tmp = score_tmp + s
    # print("scored word:", scored_words)
    return score_tmp


if __name__ == "__main__":
    input_list = []
    print(grid)
    print("Start typing your words! (press enter after each word and enter 'X' when done):")
    while True:
        user_input = input()
        if user_input == "X":
            break
        else:
            input_list.append(user_input)
    score = boggle(grid, input_list)
    print("Your total score is", score, "points!")
