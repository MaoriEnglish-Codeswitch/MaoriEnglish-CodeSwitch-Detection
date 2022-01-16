### Example of training fastText embeddings.

./fasttext cbow -input bilingual-text-input.txt -output Embeddings-300 -dim 300 -ws 5 -neg 10 -minn 5 -maxn 5; # option I CBOW

./fasttext skipgram -input bilingual-text-input.txt -output Embeddings-300SG -dim 300 -ws 5 -neg 10 -minn 5 -maxn 5; # option II Skipgram