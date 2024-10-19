import os

# Init : error corpus, tags & words by line, start tags, number of sentences
error_count = 0 
tagset = [] 
wordset = [] 
start_tag = {} 
startlen = 0 

# train file
file = open('./dataset/tagged_train.txt', "r",  encoding="utf8")

for index, line in enumerate(file) :
    """
    preprocessing
    -- input : sentence-ID word/tag word/tag word/tag ... by line
    -- output 
    ---- tagset : tag tag tag... by line (2D list)
    ---- wordset : word word word ... by line (2D list)
    """

    startlen = index+1 # count number of sentences(lines)
    sentence = line.split()
    tag_temp = [] 
    word_temp = []

    for i in range(len(sentence)) :
        # skip sentence-id
        if i == 0 :
            continue 
            
        # wrong sample : not in '/'
        if '/' not in sentence[i] :
            error_count += 1
            continue

        words = sentence[i].split('/')

        # wrong sample : "more than two '/' in words"
        if len(words) != 2 :
            error_count += 1
            continue

        word, tag = words[0], words[1]
        tag_temp.append(tag)
        word_temp.append(word)

        # count start tags
        if i == 1 :
            if tag not in start_tag :
                start_tag[tag] = 1
            else :
                start_tag[tag] += 1

    tagset.append(tag_temp)
    wordset.append(word_temp)

"""
Init
-- tag_count, word_count : Count types of tags & words
-- transition_count : count transition,  [prev_tag][cur_tag] = count (2D dictionary)
-- emission_count : count emission,  [tag][word] = count (2D dictionary)
"""
tag_count = {}
word_count = {}
transition_count = {}
emission_count = {}

# count tags, words, transition, emission
for taglist, wordlist in zip(tagset, wordset) :
    for i, (tag, word) in enumerate(zip(taglist, wordlist)) :
        # transition count (if i == 0 , start tag )
        if i != 0 : 
            prev_tag = taglist[i-1]
            
            if prev_tag not in transition_count :
                transition_count[prev_tag] = {}
            if tag not in transition_count[prev_tag] :
                transition_count[prev_tag][tag] = 1
            else :
                transition_count[prev_tag][tag] += 1
            
        # emisiion count
        if tag not in emission_count :
            emission_count[tag] = {}
        if word not in emission_count[tag] :
            emission_count[tag][word] = 1
        else :
            emission_count[tag][word] += 1

        # tag & word count
        if tag not in tag_count :
            tag_count[tag] = 1
        else :
            tag_count[tag] += 1
        
        if word not in word_count :
            word_count[word] = 1
        else :
            word_count[word] += 1

##### addingone smoothing ######

# adding-one smoothing - start tag
for tag in tag_count.keys() :
    if tag not in start_tag :
        start_tag[tag] = 1
    else :
        start_tag[tag] += 1

# adding-one smoothing - transition 
for key, dict in transition_count.items():
    # [tag][tag] --> +1
    for key2 in dict.keys() : # 
        transition_count[key][key2] += 1

    # Not exist [tag][tag] --> 1
    for tag in tag_count.keys():
        if tag not in transition_count[key]:
            transition_count[key][tag] = 1

# Not exist [tag] --> add [tag] & [tag][all_tags] = 1
for tag in tag_count.keys() :
    if tag not in transition_count.keys():
        transition_count[tag] = {}
        for tag2 in tag_count.keys():
            transition_count[tag][tag2] = 1


# adding-one smoothing - emission
for key, dict in emission_count.items():
    for word in word_count.keys():
        if word in emission_count[key] :
            emission_count[key][word] += 1
        else :
            emission_count[key][word] = 1


# Init probabilities
transition_prob = {}
emission_prob = {}
start_prob = {}

# Calculate start_prob
# +len(tag_count) : addingone_smoothing
for tag in start_tag.keys():
    start_prob[tag] = start_tag[tag] / (startlen + len(tag_count))


# transition prob with addingone
# +len(tag_count) : addingone_smoothing
for prev_tag in tag_count.keys() :
    transition_prob[prev_tag] = {}
    for cur_tag in tag_count.keys() :
        transition_prob[prev_tag][cur_tag] = transition_count[prev_tag][cur_tag] / (tag_count[prev_tag] + len(tag_count))

# emission prob with add-one
# +len(word_count) : addingone_smoothing
for tag in tag_count.keys() :
    emission_prob[tag] = {}
    for word in word_count.keys() :
        emission_prob[tag][word] = emission_count[tag][word] / (tag_count[tag] + len(word_count))


# Write result 
with open('./model/HMM_addingone_model.dat', 'w', encoding="utf-8") as file:
    file.write("transition_prob\n")
    for prev_tag, dict in transition_prob.items():
        file.write(f"{prev_tag} ")
        for cur_tag, prob in dict.items() : 
            file.write(f"{cur_tag} {prob} ")
        file.write("\n")

    file.write("\nemission_prob\n")
    for tag, dict in emission_prob.items():
        file.write(f"{tag} ")
        for word, prob in dict.items():
            file.write(f"{word} {prob} ")
        file.write("\n")

    file.write("\nstart_prob\n")
    for key, value in start_prob.items():
        file.write(f"{key} {value}\n")




