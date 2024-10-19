import argparse
import numpy as np
import sys
import os
import random

# Use Console (defalut)
# python viterbi.py

# Use File
# python viterbi.py --input ./dataset/tagged_test.txt --output_dir result --output Viterbi_tagging.txt

# Use another model   
# python viterbi.py --model another_model.dat

# Use another test file
# python viterbi.py --input ./..testfile_path/testfile_name.extension --output_dir result --output Viterbi_tagging.txt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="./model/HMM_addingone_model.dat", help="test file name")
parser.add_argument('--input', type=str, default="None", help="test file name")
parser.add_argument('--output_dir', type=str, default="None", help="output directory name")
parser.add_argument('--output', type=str, default="None", help="output file name")
args = parser.parse_args()

def test_preprocess(file):
    """
    Test dataset is obtained automatically by applying Stanford's POS tagger on a publicly available English Raw Corpus.
    Check the tagged_test.txt file in the dataset folder.

    Line format : sentence-ID word/tag word/tag word/tag .....

    preproecessing 
    -- input : tagged_test.txt
    -- output 
    ---- tagset : tag per line : tagset[line_index][sentence_index] = tag
    ---- wordset : word per line : wordsest[line_index][sentence_index] = word
    ---- return : tagset, wordset (2D list)
    """

    tagset = []
    wordset = []
    
    for line in file :
        sentence = line.split()
        tag_temp = [] 
        word_temp = []

        for i in range(len(sentence)) :
            if i == 0 :
                continue 
                
            # wrong sample : not in '/'
            if '/' not in sentence[i] :
                continue

            words = sentence[i].split('/')

            # wrong sample : "more than two '/' in words"
            if len(words) != 2 :
                continue

            word, tag = words[0], words[1]
            tag_temp.append(tag)
            word_temp.append(word)

        tagset.append(tag_temp)
        wordset.append(word_temp)
    
    return tagset, wordset

def model_preprocess(model) :
    """
    Use HMM_addingone_model.dat trained by tagged_train.txt

    HMM_addingone_model.dat format
    -- transition_prob : [y_prev][y_cur] = probability (2D dictionary)
    ---- PREV_TAG NEXT_TAG Prob NEXT_TAG Prob...
    -- emission_prob : [y_cur][x_cur] = probability (2D dictionary)
    ---- TAG WORD Prob WORD Prob WORD Prob...
    -- start_prob : [tag] = probability (1D dictionary)
    ---- TAG Prob 

    preproecessing 
    -- input : HMM_addingone_model.dat
    -- output : transition_probabilities, emission_probabilities, start_probabilities (2D, 2D, 1D dictionary)
    """

    transition = {}
    emission = {}
    start = {}

    cur_dict = None
    
    for line in model:
        if line.strip() == "":
            continue

        cur_line = line.split()
        
        if len(cur_line) == 1 :
            if cur_line[0] == "transition_prob" :
                cur_dict = transition
            elif cur_line[0] == "emission_prob" :
                cur_dict = emission
            elif cur_line[0] == "start_prob" :
                cur_dict = start
            continue

        if cur_dict == transition or cur_dict == emission:
            prev = None
            for index, value in enumerate(cur_line) :
                if index == 0 :
                    prev = value
                    cur_dict[prev] = {}
                    continue

                if index % 2 == 0 :
                    cur_dict[prev][cur_line[index-1]] = float(value)
        elif cur_dict == start :
            key, value = cur_line[0], cur_line[1]
            cur_dict[key] = float(value)
    
    taglist = [] 
    for key in start.keys():
        taglist.append(key)

    wordlist = []
    for key in emission[taglist[0]] :
        wordlist.append(key)

    return transition, emission, start, taglist, wordlist

def Viterbi_file(transition, emission, start, taglist, wordlist):
    """
    transition, emission, start : Precomputed transition, emission, start probablities
    taglist, wordlist : Types of all tags & words

    Use Viterbi algorithm
    -- Use log to prevent cacluation errror
    -- Use randomly choose from the wordlist, if UNK words appear (UNK : words that appear in test but not in train)
    
    Outputs the predicted POS tagging to the output file(args.output)
    Return : 
    -- pred_tag list : predicted tag lists for each line
    -- GT_tag list : GT tag lists for each line
    """

    infile = open(args.input, "r",  encoding="utf8") 
    GT_tag, corpus = test_preprocess(infile)
    t,e,s = transition, emission, start

    total = len(corpus) # total number of  sentences
    pred_tag = []

    for counting, Sentence in enumerate(corpus) :
        # testing
        sys.stdout.write(f"\rTest : {counting+1} / {total}") 

        # State Latice & backpath
        state = {} 
        backstep = {} 
        tagpath = []

        # Calculate start prob 
        for index in range(0, len(Sentence)) :
            state[index] = {}
            if index == 0:
                # Unknown word --> random choice in wordlist
                word = Sentence[0]
                if word not in wordlist : 
                    word = random.choice(wordlist)
            for tag in taglist :
                if index == 0 :
                    state[0][tag] = np.log(s[tag]) + np.log(e[tag][word])
                    backstep[0,tag] = "None"
                else :
                    state[index][tag] = 0

        # Calculate state prob
        for index in range(1, len(Sentence)) :
            word = Sentence[index]
            # Unknown word --> random choice in wordlist
            if word not in wordlist : 
                word = random.choice(wordlist)

            for cur_tag in taglist :
                emission_prob = np.log(e[cur_tag][word]) 

                for prev_tag in taglist :
                    cur_prob = state[index-1][prev_tag] + np.log(t[prev_tag][cur_tag]) + emission_prob

                    if state[index][cur_tag] == 0 :
                        state[index][cur_tag]  = cur_prob
                        backstep[index,cur_tag] = prev_tag
                    else :
                        if state[index][cur_tag] < cur_prob :
                            state[index][cur_tag] = cur_prob
                            backstep[index,cur_tag] = prev_tag
                    
        # backpath
        tag = max(state[len(Sentence)-1], key=state[len(Sentence)-1].get)

        for index in range(len(Sentence)-1, -1, -1) :
            tagpath.insert(0,tag)
            tag = backstep[index, tag]

        pred_tag.append(tagpath)
    
    print()
    infile.close()

    # Write Output file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, args.output), 'w', encoding="utf-8") as file:
        for i in range(0, len(corpus)):
            file.write(f"Sentence #{i} : ")
            for j in range(0, len(corpus[i])):
                file.write(f"{corpus[i][j]} ")
            file.write("\n")

            file.write(f"POS tagging #{i} : ")
            for j in range(0, len(pred_tag[i])):
                file.write(f"{pred_tag[i][j]} ")
            file.write("\n")

    return GT_tag, pred_tag

def Viterbi_Console(transition, emission, start, taglist, wordlist):
    """
    Input: Sentences separated by blank lines (Enter '0' to exit)
    Output: Tag column
    """

    t,e,s = transition, emission, start
    print("Enter sentences separated by spaces on the console. Press 0 to exit")

    while True :
        # readline
        line = list(sys.stdin.readline().rstrip().split()) 
       
        # terminate
        if len(line) == 1 and line[0] == "0":
            break

        state = {} 
        backstep = {} 
        tagpath = []

        for index in range(0, len(line)) :
            state[index] = {}
            if index == 0:
                # UNK
                word = line[0]
                if word not in wordlist : 
                    word = random.choice(wordlist)
            for tag in taglist :
                if index == 0 :
                    state[0][tag] = np.log(s[tag]) + np.log(e[tag][word])
                    backstep[0,tag] = "None"
                else :
                    state[index][tag] = 0
        
        for index in range(1, len(line)) :
            word = line[index]
            # UNK
            if word not in wordlist : 
                word = random.choice(wordlist)

            for cur_tag in taglist :
                emission_prob = np.log(e[cur_tag][word]) 

                for prev_tag in taglist :
                    cur_prob = state[index-1][prev_tag] + np.log(t[prev_tag][cur_tag]) + emission_prob

                    if state[index][cur_tag] == 0 :
                        state[index][cur_tag]  = cur_prob
                        backstep[index,cur_tag] = prev_tag
                    else :
                        if state[index][cur_tag] < cur_prob :
                            state[index][cur_tag] = cur_prob
                            backstep[index,cur_tag] = prev_tag

        tag = max(state[len(line)-1], key=state[len(line)-1].get)

        for index in range(len(line)-1, -1, -1) :
            tagpath.insert(0,tag)
            tag = backstep[index, tag]

        for tagging in tagpath :
            sys.stdout.write(f"{tagging} ")
        
        print()

def Evalution(ground_truth, predict) :
    """
    Evaluation 
    -- gt, pred : GT Tag, Predict Tag
    -- total, correct, wrong, accuracy : Nubmer of total words, correct tagging, wrong tagging
    -- Accuracy : correct / total (%)
    """

    gt, pred = ground_truth, predict

    total, correct, wrong, accuracy = 0, 0, 0, 0 # correct, wrong

    for i in range(0, len(pred)):
        for j in range(0, len(pred[i])):
            total += 1
            if gt[i][j] == pred[i][j] :
                correct += 1
            else :
                wrong += 1

    accuracy = round(correct/total, 2) * 100

    sys.stdout.write(f"Result : [Total words : {total}] [Correct words : {correct}] [Wrong words : {wrong}] [Accuracy : {accuracy}%]")
    
model = open(args.model, "r", encoding="utf8") # pre-train model
transition ,emission ,start_prob, taglist, wordlist = model_preprocess(model) # transition, emission, start 

if args.input == "None": # Console
    Viterbi_Console(transition ,emission ,start_prob, taglist, wordlist)
else : # File
    gt, pred = Viterbi_file(transition ,emission ,start_prob, taglist, wordlist)
    Evalution(gt, pred)



    




    
        