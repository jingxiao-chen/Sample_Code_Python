########################################
##########                    ##########
########## PH 1975 Homework 3 ##########
##########                    ##########
########################################
## Jingxiao Chen, Fangyu Li, Sepideh Saroukhani
##########   Sep. 29, 2019    ##########
########################################
######      Caesar Cipher        #######
########################################

import sys
import string

plaintext_upper = string.ascii_uppercase


def mix_alphabet(keyword_input):
    keyword_unique = ""
    for i in range(0, len(keyword_input)):
        if keyword_input.upper()[i] not in keyword_unique:
            keyword_unique = keyword_unique + keyword_input.upper()[i]
    output = "" + keyword_unique
    for j in range(0, len(plaintext_upper)):
        if plaintext_upper[j] not in keyword_unique:
            output = output + plaintext_upper[j]
    return output


def caesar_cipher(text, mixed_alphabet, shift=3):
    print("input text:", text)
    ciphered = ""
    for i in range(0, len(text)):
        char = text[i]
        if char.isupper():
            ind = mixed_alphabet.index(char)
            ciphered = ciphered + mixed_alphabet[(ind + shift) % 26]
        elif char.islower():
            ind = mixed_alphabet.lower().index(char)
            ciphered = ciphered + mixed_alphabet.lower()[(ind + shift) % 26]
        else:
            ciphered = ciphered + char
    return ciphered


if __name__ == "__main__":
    filename = sys.argv[1]
    with open(filename, "r") as input_file:
        input_text = input_file.read()
    keyword = input("Please enter a keyword for the " + filename.rsplit('.', 1)[0] + " cipher: ")

    print("Plain text:", plaintext_upper)
    ciphertext = mix_alphabet(keyword)
    print("Cipher text:", ciphertext)
    ciphered_text = caesar_cipher(input_text, ciphertext, shift=3)
    print("The ciphered content: ", ciphered_text)
