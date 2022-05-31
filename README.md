
## Thesis on Micropost-incident-detection

## Running instructions:

1. Install dependencies from conda environment "env.yml"
2. Download word embeddings using "Data preparation.ipynb"
3. Experiments notebooks are available in "Notebooks/Experiments" folder
4. Results: CSV files and plots are present in "results" folder

## Abstract
Social media is becoming the most powerful medium for sharing information, like incidents. Due to its up-to-date real-time information, social media has been exploited as a real-time data source, especially Twitter, for text-based tasks. These Twitter posts related to incidents can be used for building early warning systems for hyper-local and larger-scale incidents and crises. However, due to the large number of posts every day, it is impractical to examine these posts manually. Because of this, automatic incident detection is necessary to investigate only incident-related data. One way of solving incident detection is using word embedding based Convolutional Neural Networks(CNN). Related research indicates that the word embedding-based CNN model  performed well on text classification tasks. Later, researchers added a BiGRU before the output layer to the CNN model to capture long-term dependencies between tokens, which produced good results in offensive language incident detection. First, we examine word embedding based CNN performance on micropost incident detection, and then we evaluate the influence of adding BiGRU to CNN on incident detection. 
Furthermore, different types of word embedding will be employed to evaluate how domain-specific word embeddings perform on incident detection tasks. Finally, we will evaluate the impact of word embeddings on incident detection in different regions having different colloquial languages and contexts. We used three different datasets which have different modalities to solve our research problems in this thesis. Also, we used different word embeddings, which are generated on different corpora with varying vocabulary size. The result we achieved in the thesis suggested that CNN and CNN with BiGRU are not suitable for the incident detection task. Furthermore, word embeddings generated on different corpora with varying vocabulary size have no effect on incident detection. Finally, word embeddings aid in the generalization of incident detection across regions with different colloquial languages and contexts. They are, however, not significantly better than the token-based machine learning model.

## References

 - Yoon Kim. “Convolutional Neural Networks for Sentence Classification”. In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Doha, Qatar: Association for Computational Linguistics, Oct. 2014, pp. 1746–1751. doi: 10.3115/v1/D14-1181. url: https://aclanthology.org/D14-1181
 - S. Urchs, L. Wendlinger, J. Mitrović and M. Granitzer, "MMoveT15: A Twitter Dataset for Extracting and Analysing Migration-Movement Data of the European Migration Crisis 2015," 2019 IEEE 28th International Conference on Enabling Technologies: Infrastructure for Collaborative Enterprises (WETICE), 2019, pp. 146-149, doi:10.1109/WETICE.2019.00039.

- Jelena Mitrović, Bastian Birkeneder, and Michael Granitzer. “nlpUP at
SemEval-2019 Task 6: A Deep Neural Language Model for Offensive Language Detection”. In: Jan. 2019, pp. 722–726. doi:10.18653/v1/S19-2127

- Jeffrey Pennington, Richard Socher, and Christopher D. Manning. “GloVe: Global Vectors for Word Representation”. In: Empirical Methods in Natural Language Processing (EMNLP). 2014, pp. 1532–1543. url:[http://www.aclweb.org/anthology/D14-1162](http://www.aclweb.org/anthology/D14-1162)

- Fréderic Godin. Twitter Word2vec and FastText word embeddings. Aug. 17, 2019. url: [https://web.archive.org/web/20200922132606/https://fredericgodin.com/research/twitter-word-embeddings/](https://web.archive.org/web/20200922132606/https://fredericgodin.com/research/twitter-word-embeddings/)
