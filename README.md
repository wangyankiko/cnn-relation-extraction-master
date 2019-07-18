# Convolutional Neural Networks for Relation Extraction

Tensorflow Implementation of Deep Learning Approach for Relation Extraction Challenge([**SemEval-2010 Task #8**: *Multi-Way Classification of Semantic Relations Between Pairs of Nominals*](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)) via Convolutional Neural Networks.

<p align="center">
	<img width="700" height="400" src="https://user-images.githubusercontent.com/15166794/32838125-475cbdba-ca53-11e7-929c-2e27f1aca180.png">
</p>


## Usage
### Train
* Train data is located in "*<U>SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT*</U>".
* "[GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/)" is used as pre-trained word2vec model.
* Performance (accuracy and f1-socre) outputs during training are **NOT OFFICIAL SCORE** of *SemEval 2010 Task 8*. To compute the official performance, you should proceed the follow [Evaluation](#evaluation) step with checkpoints obtained by training.

##### Display help message:
```bash
$ python train.py --help
```
##### Train Example:
```bash
$ python train.py --embedding_path "GoogleNews-vectors-negative300.bin"
```

### Evaluation
* You can get an **OFFICIAL SCORE** of *SemEval 2010 Task 8* for test data by following this step. [README](SemEval2010_task8_all_data/SemEval2010_task8_scorer-v1.2/README.txt) describes how to evaluate the official score.
* Test data is located in "<U>*SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT*</U>".
* **MUST GIVE `--checkpoint_dir` ARGUMENT**, checkpoint directory from training run, like below example.

##### Evaluation Example:
```bash
$ python eval.py --checkpoint_dir "runs/1523902663/checkpoints/"
```


## Results
#### Officiail Performance
![performance](https://user-images.githubusercontent.com/15166794/47507952-24510a00-d8ae-11e8-93e1-339e19d0ab9c.png)

#### Learning Curve (Accuracy)
![acc](https://user-images.githubusercontent.com/15166794/47508193-988bad80-d8ae-11e8-800c-4f369cf23d35.png)

#### Learning Curve (Loss)
![loss](https://user-images.githubusercontent.com/15166794/47508195-988bad80-d8ae-11e8-82d6-995367bc8f42.png)


## SemEval-2010 Task #8
* Given: a pair of *nominals*
* Goal: recognize the semantic relation between these nominals.
* Example:
	* "There were apples, **<U>pears</U>** and oranges in the **<U>bowl</U>**." 
		<br> → *CONTENT-CONTAINER(pears, bowl)*
	* “The cup contained **<U>tea</U>** from dried **<U>ginseng</U>**.” 
		<br> → *ENTITY-ORIGIN(tea, ginseng)*


### The Inventory of Semantic Relations
1. *Cause-Effect(CE)*: An event or object leads to an effect(those cancers were caused by radiation exposures)
2. *Instrument-Agency(IA)*: An agent uses an instrument(phone operator)
3. *Product-Producer(PP)*: A producer causes a product to exist (a factory manufactures suits)
4. *Content-Container(CC)*: An object is physically stored in a delineated area of space (a bottle full of honey was weighed) Hendrickx, Kim, Kozareva, Nakov, O S´ eaghdha, Pad ´ o,´ Pennacchiotti, Romano, Szpakowicz Task Overview Data Creation Competition Results and Discussion The Inventory of Semantic Relations (III)
5. *Entity-Origin(EO)*: An entity is coming or is derived from an origin, e.g., position or material (letters from foreign countries)
6. *Entity-Destination(ED)*: An entity is moving towards a destination (the boy went to bed) 
7. *Component-Whole(CW)*: An object is a component of a larger whole (my apartment has a large kitchen)
8. *Member-Collection(MC)*: A member forms a nonfunctional part of a collection (there are many trees in the forest)
9. *Message-Topic(CT)*: An act of communication, written or spoken, is about a topic (the lecture was about semantics)
10. *OTHER*: If none of the above nine relations appears to be suitable.


### Distribution for Dataset
* **SemEval-2010 Task #8 Dataset [[Download](https://drive.google.com/file/d/0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk/view?layout=list&ddrp=1&sort=name&num=50#)]**

	| Relation           | Train Data          | Test Data           | Total Data           |
	|--------------------|:-------------------:|:-------------------:|:--------------------:|
	| Cause-Effect       | 1,003 (12.54%)      | 328 (12.07%)        | 1331 (12.42%)        |
	| Instrument-Agency  | 504 (6.30%)         | 156 (5.74%)         | 660 (6.16%)          |
	| Product-Producer   | 717 (8.96%)         | 231 (8.50%)         | 948 (8.85%)          |
	| Content-Container  | 540 (6.75%)         | 192 (7.07%)         | 732 (6.83%)          |
	| Entity-Origin      | 716 (8.95%)         | 258 (9.50%)         | 974 (9.09%)          |
	| Entity-Destination | 845 (10.56%)        | 292 (10.75%)        | 1137 (10.61%)        |
	| Component-Whole    | 941 (11.76%)        | 312 (11.48%)        | 1253 (11.69%)        |
	| Member-Collection  | 690 (8.63%)         | 233 (8.58%)         | 923 (8.61%)          |
	| Message-Topic      | 634 (7.92%)         | 261 (9.61%)         | 895 (8.35%)          |
	| Other              | 1,410 (17.63%)      | 454 (16.71%)        | 1864 (17.39%)        |
	| **Total**          | **8,000 (100.00%)** | **2,717 (100.00%)** | **10,717 (100.00%)** |



## Reference
* **Relation Classification via Convolutional Deep Neural Network** (COLING 2014), D Zeng et al. **[[review]](https://github.com/roomylee/paper-review/blob/master/relation_extraction/Relation_Classification_via_Convolutional_Deep_Neural_Network.md)** [[paper]](http://www.aclweb.org/anthology/C14-1220)
* **Relation Extraction: Perspective from Convolutional Neural Networks** (NAACL 2015), TH Nguyen et al. **[[review]](https://github.com/roomylee/paper-review/blob/master/relation_extraction/Relation_Extraction-Perspective_from_Convolutional_Neural_Networks.md)** [[paper]](http://www.cs.nyu.edu/~thien/pubs/vector15.pdf)
* dennybritz's cnn-text-classification-tf repository [[github]](https://github.com/dennybritz/cnn-text-classification-tf)


