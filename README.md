# Projecide SQuAD - An Hybrid Question Answering System
Question Answering + Information Retrieval project based on SQuAD Dataset. **Check the pdf report for details**.

We  aim  at  building a Question Answering NLP tool based on SQuAD Dataset. Moreover we introduced an Information 
Retrieval module which allows the user to perform this preliminary step in case question and contexts are not already
coupled or, for example, a better answer to a question can be found in a different context.

We have built and evaluated the two modules (QA and IR) independently, they can be examined in folder 'base_project'
and folder 'ir_module' respectively. Then all the pre-trained models and pre-computed data are wrapped in a new 
software structure, contained in the 'compute_answers_files' folder, where the user can access all the functionality 
from 'compute_answers.py'.

Despite only this last indicated folder is needed to run the project, the dev-modules contain much more functionalities
and details.


## Table of Contents

* [About the Project](#about-the-project)
* [Prerequisites](#prerequisites)
* [Usage](#usage)
* [Conclusions](#conclusions)
* [Authors](#authors)


## About The Project
Question Answering and Information Retrieval are well-known problems, but they still push the research in the NLP field 
nowadays. Our choice is to implement and hybrid different Neural model techniques for QA and post-process the output
with more standard algorithms based on similarity. For IR, or 'context retrieval' to be more precise in our case, we 
preferred a more traditional - but very effective - approach based on TF-IDF combined with a Cosine Similarity or with
the Jensen-Shannon distance (user's choice). In this problem neural networks have shown poor capabilities probably due
to a poor document embedding architecture, limited by the hardware of a Home PC.

The code and the virtual environment are set up to automatically to exploit CUDA whenever it is possible thanks to the
tensorflow-gpu toolchain.

Custom developed architecture:

![neural_model](https://user-images.githubusercontent.com/38500291/203978597-b0d927f8-7d1d-47c9-8895-dafa2dc8f19f.PNG width="60")


## Prerequisites
The following python packages have to be installed on the machine in order to run our 
implementation of the models:
* pandas~=1.3.2
* numpy~=1.20.3
* gensim~=4.1.2
* tensorflow~=2.7.0
* tqdm~=4.62.3
* nltk~=3.6.6
* tensorflow-gpu~=2.7.0 (highly recommended for good predictions time performances)
* matplotlib~=3.4.3
* keras~=2.7.0
* scikit-learn~=1.0.2
* scipy~=1.7.0
* spacy~=3.2.1


## Usage
The projecide_squad folder contains an already set up virtual-env Python 3.9 with all the dependency and libraries 
included. It's the easiest plug-and-play configuration we figured it out. For the pre-trained model and pickle archives, contact us.

### Setup the venv
Activate venv
```console
$ cd <path to projecide_squad>\venv\Scripts
```
```console
$ activate.bat
```
Now '(venv)' should appear. Return to the root folder.
```console
$ cd..
$ cd..
```
```console
<path to projecide_squad>$ cd compute_answers_files
```
### Run the project
```console
$ python compute_answer.py [-in INPUT_FILE]{"unanswered_example.json"} [-dr MODE]{0} [-cr TOP_K]{5}
```
This command will return a .txt file in ```/compute_answers_files/predicitons/out/```.

#### Arguments meaning

- ```--inputfile -in INPUT_FILE``` - name of the input file (included .json) formatted as SQuAD dataset (with or without
  answers). This file *MUST* be in ```/compute_answers_files/predicitons/in/```. Accepts UTF-8 or W-1252 encoding. 
  Default ```"unanswered_example.json"```
- ```--docretrieval -dr MODE``` - integer in [0, 1, 2]: 0 - no Information Retrieval, 1 - IR + Cosine Similarity, 2 - 
  IR + Jensen Shannon distance. Cosine Similarity is much faster but less accurate. Dafault 0.
- ```--confidenceretrieval -cr TOP_K``` - integer in [1, 10]: indicates the number of TOP_K contexts to keep in the 
document retrieval. If Jensen Shannon is active, we suggest no more than 4. In case of Cosine Similarity at least 3. 
  Default 5.

## Conclusions
The files ```/base_project/src/master_qa.py``` and  ```/ir_module/src/master_ir.py``` manage respectively the developing
phase of Question Answering and Information Retrieval. From the global flag is possible to setup new training or create
new datasets splits and cleaning. The user code in ```/compute_answers_files/``` contains everything is needed to 
conduce the experiments with the best settings we have found. 

Folder ```/evaluate_methods/``` contains the ```evaluate.py```, the official SQuAD evaluation script. In the same
folder are included also the test set data (original and predict) to use it.
```
QA task, top-4 proposals, results:

"exact": 1.24167513263348,
"f1": 23.465646450806872,
"total": 8859,
"HasAns_exact": 1.24167513263348,
"HasAns_f1": 23.465646450806872,
"HasAns_total": 8859

```

Several hyper parameters could be change in the dev-code with a more powerful hardware.

## Authors
* [**Alessandro Maggio**](alessandro.maggio5@studio.unibo.it)
* [**Irene Rachele Lavopa**](irenerachele.lavopa@studio.unibo.it)
* [**Lorenzo Loschi**](lorenzo.loschi@studio.unibo.it)
