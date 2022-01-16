# Māori-English Code Switch Detection
A repository containing language models and training scripts for language detection of te reo Māori, and Māori-English Code-switching detection. 

## Table of Contents
* [General Information](#general-information)
* [Data](#data)
* [Sources](#sources)
* [Kaitiakitanga License and how to cite us](#kaitiakitanga-license-and-how-to-cite-us)

## General Information
We include the following in this repository:

|No.| Item | Availability |
|---| ------------- | ------------- |
| 1.| Training code for word embeddings.  | Available in this repo.  |
| 2.| Sample code for language detection and code-switching detection. | Available in this repo.  | 
| 3.| Trained language models for Code-switching detection.| Available in this repo.  | 
| 4.| Demo: Language Detection and code-switching detection using trained models.| Available in this repo.  | 
| 5.| Pre-trained Monolingual Māori and Bilingual Māori-English word embeddings. | Available upon request. Contact xx@yy.com | 

## Data

|Data | Information | Text | Labels | 
|---| ------------- | ------------- |  ------------- |    
|[Hansard data](https://www.parliament.nz/en/pb/hansard-debates/rhr/) | 2,021,261 sentences & 36,757,230 words | formal language | word-level & sentence level language labels |
|[MLT corpus](https://kiwiwords.cms.waikato.ac.nz/corpus/) | 2,500 sentences & 50,000 words | informal language | tweet level labels: relevance/irrelevance |
|[RMT corpus](https://kiwiwords.cms.waikato.ac.nz/rmt_corpus/) | 79,018 sentences & 1,000,000 words | informal language | Māori words are identified and labelled |

## Demos: Examples in Jupiter Notebook
### Language Detection 

1. [Using Google Translate for the Hansard data](Language-Detection/google_trans_demo.ipynb)
2. [Using BiLSTM for RMT corpus](Language-Detection/RMT_corpus_language_detection.ipynb)

### Code-switching Detection



## Sources
The initial developments in this work have been inspired by the work by Te Hiku media - https://github.com/TeHikuMedia/nga-tautohetohe-reo

## Kaitiakitanga License and how to cite us
(based on https://github.com/TeHikuMedia/Kaitiakitanga-License/blob/tumu/LICENSE.md)
All the models and embeddings developed here are bound by the Kaitiakitanga Lincense

**Preamble**

Kaitiaki is a Māori word without specific English translation, but its meaning is similar to the words guardian, protector, and custodian . In this context we protect the code in this repository and will provide access to the embeddings as we deem fit through our tikanga (Māori customs and protocols).

While we recognize the importance of open source technology, we are mindful that the majority of tangata whenua and other indigenous peoples may not have access to the resources that enable them to benefit from open source technologies. As tangata whenua, our ability to grow, develop, and innovate has been stymied through colonization. We must protect our ability to grow as tangata whenua. By simply open sourcing our data and knowledge, we further allow ourselves to be colonised digitally in the modern world.

The Kaitiakitanga License is a work in progress. It's a living license. It will evolve as we see fit. We hope to develop a license that is an international example for indigenous people's retention of mana over data and other intellectual property in a Western construct.

**Terms** 

You must contact us and seek permission to access, use, contribute towards, or modify code in this repository;
You may not use code in this repository or any derivations for commercial purposes unless we explicitly grant you the right to do so;
All works derived from code in this repository are bound by the Kaitiakitanga License;
All works that make use of any code in this repository are bound by the Kaitiakitanga License.
You must contact us to obtain access to the word embeddings and to be used in your projects;

We acknowledge that the research team consist of Māori and non-Māori researchers, and the non-Māori researchers share the views of the Māori researchers, thereby remaining kaitiaki of the data and resources that have been shared with them.

**Upon using the resources in this repository or embeddings upon request, please cite us as:**
Author 1, Author 2, Author 3, "Language Models for Code-switch Detection of te reo Māori and English in Low-resource Setting", xx conference, 2022.

