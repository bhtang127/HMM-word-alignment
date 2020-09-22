## Requirement:

python >= 3.6.x

numpy >= 1.0.x

tqdm >= 4.x


## Usage:

run python3 word_align.py -h to see help information

usage: word_align.py [-h] [-d data_prefix] [-e English_suffix]
                     [-f French_suffix] [-n sentences_number] [-m model]
                     [-i max_iter] [-s smooth_level] [--rt] [--rs]

Word Alignment

|    option     | description | 
| -------------:| -----------:|
|  -h, --help   | show this help message and exit |
|  -d           | Data filename prefix (default=data/hansards) |
|  -e           | Suffix of English filename (default=e) |
|  -f           | Suffix of French filename (default=f) |
|  -n           | Number of sentences to use for training and alignment |
|  -m, --model  |  Abbrev for models. Only support 'IBM1' for IBM model 1 and 'HMM' for HMM-based alignment model. HMM will be initialized using IBM1 (default='HMM') |
|  -i, --iter   | Max iteration number for iterative models. (default=5) |
|  -s, --smooth | Smooth level for prediction in HMM, 0.4 is a good value in practice (default=0.4) |
|  --rt         | Training models use transposed French and English text. E-F will be trained instead of F-E if using this option |
|  --rs         | Training models while each sentence is in reverse order |


python3 combine.py align1.a align2.a align3.a ...

combine serval alignment by chosen pairs appeared over half of all alignments

python3 refine.py align1.a align2.a

refine align1.a by align2.a

## Example

_train an IBM model 1 with 5 iteration on first 1000 sentences_

`python3 word_align.py -m IBM1 -n 1000 --iter 5 > align1.a`

_train an HMM model on all data but transpose English and French sentences_

`python3 word_align.py --rt > align2.a`

_combine above 2 alignments_

`python3 combine.py align1.a align2.a > align3.a`

_refine align2.a with align1.a_

`python3 refine.py align2.a align1.a > align4.a`


## Codes that generate our final alignment

```
python3 word_align.py -m IBM1 > ibm.a
python3 word_align.py -m IBM1 --rt > ibm_rt.a
python3 word_align.py > hmm.a
python3 word_align.py --rt > hmm_rt.a
python3 word_align.py --rs > hmm_rs.a
python3 word_align.py --rt --rs > hmm_rt_rs.a
python3 combine.py ibm.a ibm_rt.a > ibm_int.a
python3 combine.py hmm.a hmm_rt.a hmm_rs.a hmm_rt_rs.a > hmm_int.a
python3 refine.py hmm_int.a ibm_int.a > alignment
```

we also hold this pipeline in file align, usage:

`python3 align -n num_sentences > alignment`

This will run above pipeline with top num_sentences pair of sentences. And output to file alignment

**Notice**: since speed issue default -n for align is 100. Use
`python3 align -n 100000 > alignment` 
can reproduce our results. The whole process will take about 4 hours.

if everything goes right it should have Precision = 0.964371; Recall = 0.843195; AER = 0.089592 on first 37 sentences
