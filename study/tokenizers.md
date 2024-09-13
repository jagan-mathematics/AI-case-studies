**Finding 1: WordPiece Tokenization Improves NMT Performance**

NMT models often struggle with rare words or words not found in their vocabulary (OOV). This can significantly impact translation quality. A key finding is that using a WordPiece tokenizer helps address this issue. 

**Here's how WordPiece tokenization benefits NMT:**

* **Breaks down rare words:** By splitting infrequent words into smaller subword units, the model encounters these units more frequently during training. This improves its ability to translate them accurately.
* **Reduces vocabulary size:**  Instead of needing entries for every possible word, the model learns a smaller set of subword units that can be combined to represent most words. This can lead to more efficient training and potentially better generalization.

**Evidence:** This approach was used in a Google production NMT model and achieved state-of-the-art results at the time[1]. 

### Limitation of Word Tokenizer
Word-level tokenizers are relatively effective in processing European languages, where spaces provide clear word boundaries. However, this method is limited in languages like Chinese, which do not have clear word boundaries. Moreover, the flexible morphological inflections in language, the constant emergence of new words, and the
prevalence of spelling errors in corpora make it difficult for word-level vocabularies to generalize in practical applications.[3]




## Reference:
[1] [wordpeice tokenizer in Google production **NMT** model](https://arxiv.org/pdf/1609.08144v2)
[2] [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909)
[3] [rethinking tokenizers](https://arxiv.org/pdf/2403.00417)
[4] [Andrej Karpathy video](https://www.youtube.com/watch?v=zduSFxRajkE)
