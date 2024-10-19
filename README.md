# Viterbi Algorithm
Perform POS tagging using the Viterbi algorithm with the Adding One technique.


# Usage
Use python and numpy 


# Dataset

Created dataset using Stanford Random POS tagging.

Data Format : Codenumber word/tag word/tag ....


### tagged_dev, tagged_test.txt, tagged_train.txt
```
702c412a8e8fbfed23f7db1f9e3ae5d9bae3e7cc::0	KENNEDY/NNP SPACE/NNP CENTER/NNP ,/, Florida/NNP (/( CNN/NNP )/) --/: The/DT space/NN shuttle/NN Atlantis/NNP
42c027e4ff9730fbb3de84c1af0d2c506e41c3e4::0	LONDON/NNP ,/, England/NNP (/( Reuters/NNP )/) --/: Harry/NNP Potter/NNP star/NN Daniel/NNP Radcliffe/NNP gains/NNS 
...
```

# Execution 


### Obtain bigram probabilities 
```sh
python model.py
```

### HMM_addingone_model.dat (result)
```
transition_prob
NNP NNP 0.2907522749972256 , 0.13200359283098434 ( 0.010651911552546887 ) 0.017242814338031296 : 0.010729941182998557 NN 0.06671360004439018 NNS

emission_prob
NNP LONDON 0.0008051328321982481 , 1.4719064574008192e-06 England 0.001791310158656797 ( 1.4719064574008192e-06 Reuters 5.004481955162785e-05 )

start_prob
NNP 0.1995115939898927
NP 0.14915979845714952
```


### POS tagging

- Use File
```sh
python viterbi.py --input ./dataset/tagged_test.txt --output_dir result --output Viterbi_tagging.txt
```

- Use your model
```sh
python viterbi.py --model another_model.dat
```

### Result ( Terminal & Text file )

Terminal
```sh
Test 48833 / 48833
Result : [Total words : 1082219] [Correct words : 1024412] [Wrong words : 57807] [Accuracy : 95.0%]
```

Viterbi_tagging.txt (Pos tagging results)
```
Sentence #0 : ( CNN ) -- An Irish bishop resigned Wednesday following a government report into the sexual abuse of children by Catholic clergy -- the second to do so . 
POS tagging #0 : ( NNP ) : DT JJ NN VBD NNP VBG DT NN NN IN DT JJ NN IN NNS IN NNP ) : DT JJ TO VB RB .
...
```

Sentence is the original sentence, and POS tagging is the result tagged by the model.
