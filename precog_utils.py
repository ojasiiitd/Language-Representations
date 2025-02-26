import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import torch


# get vocab_freq_dist & word_to_index
def get_corpus_info(all_words , vocab_threshold=1):
    # get unique words
    unique_words = list(set(all_words))

    # frequency distribution of the words
    print("Built frequency distribution ...")
    vocab_freq_dist = nltk.FreqDist(word for word in all_words)

    # get filtered vocab based on threshhold
    print("Filtering vocabulary ...")
    filtered_vocab = {"<UNK>"}
    for word, count in vocab_freq_dist.items():
        if count >= vocab_threshold:
            filtered_vocab.add(word)

    V = len(filtered_vocab)
    print("Vocabulary Size: " , V)

    # build word-to-index mapping
    print("Building Word-Index mapping ...")
    word_to_index = {word: idx for idx, word in enumerate(filtered_vocab)}

    # low-frequency words replaced with <UNK> in vocab frequency distribution
    print("Handling <UNK> tokens ...")
    unk_count = sum(count for word, count in vocab_freq_dist.items() if word not in filtered_vocab)
    vocab_freq_dist["<UNK>"] = unk_count

    # 10 most frequent words and their freq
    print("Most Common Words: " , vocab_freq_dist.most_common(10))

    # return filtered_vocab , vocab size , vocab freq list , word-index mapping
    return filtered_vocab , V , vocab_freq_dist , word_to_index

def get_co_occurence_matrix(V , corpus , vocabulary , word_to_index , context_window_size = 2):
    corpus = [ word for word in corpus if word in vocabulary ]

    # initialization
    co_occurrence_matrix = [[0] * V for _ in range(V)]

    # context window size
    window_size = context_window_size  # words before & after each word

    # iterate over the corpus words
    for i in tqdm(range(len(corpus))):
        target_word = corpus[i]
        target_idx = word_to_index[target_word]
        
        # context window words
        for j in range(max(0, i - window_size), min(len(corpus), i + window_size + 1)):
            if i != j:
                context_word = corpus[j]
                context_idx = word_to_index[context_word]

                co_occurrence_matrix[target_idx][context_idx] += 1

    return np.array(co_occurrence_matrix)

def get_probability_matrix(co_occurrence_matrix):
    # performing word-wise normalization
    row_sums = co_occurrence_matrix.sum(axis=1, keepdims=True)

    # handle division by zero
    row_sums[row_sums == 0] = 1

    # normalize each row
    probability_matrix = co_occurrence_matrix / row_sums

    print(probability_matrix.shape)
    return probability_matrix

# get context words
def get_context(matrix , word , vocabulary , corpus , word_to_index):
    corpus = [ word for word in corpus if word in vocabulary ]

    unique_words = list(set(corpus))
    
    context_words = []

    # mean of non zero elements
    mean_occ = np.median(matrix[word_to_index[word]][matrix[word_to_index[word]] != 0])
    
    for i,x in enumerate(matrix[ word_to_index[word] ]):
        if x > mean_occ:
            context_words.append(unique_words[i])
    return context_words

 
def reduce_dimensions_svd(matrix , dim):
    svd = TruncatedSVD(n_components=dim)
    reduced_matrix = svd.fit_transform(matrix)

    print("Smaller Matrix Shape:" , reduced_matrix.shape)

    return svd , reduced_matrix

def latent_features_cluster(svd , word_to_index):
    index_to_word = list(word_to_index.keys())

    # svd.components_ shape: (10, 30000)
    top_feature_indices = np.argsort(np.abs(svd.components_), axis=1)[:, -10:]  # Get top 10 indices per component

    # Convert indices to words
    top_words_per_component = [[index_to_word[idx] for idx in top_feature_indices[i]] for i in range(10)]

    # Print the top words per component
    for i, words in enumerate(top_words_per_component):
        print(f"Top words in Component {i+1}: {words}")

def get_smaller_matrices(probability_matrix , dim1=400 , dim2=150 , dim3=10):
    # use svd to reduce the dimensions of the embeddings and come up with Nxd matrix

    # svd_reduced , reduced_co_occurrence_matrix = reduce_dimensions_svd(probability_matrix , dim1)

    svd_compact , compact_co_occurrence_matrix = reduce_dimensions_svd(probability_matrix , dim2)
    svd_distilled , distilled_co_occurrence_matrix = reduce_dimensions_svd(probability_matrix , dim3)

    return {
        # "reduced": [svd_reduced , reduced_co_occurrence_matrix],
        "compact": [svd_compact , compact_co_occurrence_matrix],
        "distilled": [svd_distilled , distilled_co_occurrence_matrix]
        }

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return np.dot(a, b) / (norm_a * norm_b)

def batch_cosine_similarity(a, b):
    """
    Compute cosine similarity for batches of vectors using PyTorch.

    Args:
    - a: torch.Tensor of shape (n, dims)
    - b: torch.Tensor of shape (n, dims)

    Returns:
    - cosine_sim: torch.Tensor of shape (n,), containing cosine similarity for each pair
    """
    # Compute L2 norms
    norm_a = torch.norm(a, p=2, dim=1, keepdim=True)  # Shape: (n, 1)
    norm_b = torch.norm(b, p=2, dim=1, keepdim=True)  # Shape: (n, 1)

    # Avoid division by zero
    norm_a = torch.where(norm_a == 0, torch.tensor(1e-10, device=a.device), norm_a)
    norm_b = torch.where(norm_b == 0, torch.tensor(1e-10, device=b.device), norm_b)

    # Compute cosine similarity
    cosine_sim = torch.sum(a * b, dim=1) / (norm_a.squeeze() * norm_b.squeeze())  # Shape: (n,)

    return cosine_sim

def get_processed_token_idx(word , filtered_vocab , word_to_index):
    if word not in filtered_vocab:
        return word_to_index["<UNK>"]
    return word_to_index[word]

def get_fasttext_embedding(word , fasttext_model):
    fasttext_vocab = list(fasttext_model.keys())
    if word not in fasttext_vocab:
        return fasttext_model.get("unk")
    return fasttext_model.get(word)

def similar_embedding_words(fasttext_model, vector, topn=10):
    # finds top n most similar words (cosine similarity) and returns their indexes
    
    vocab = list(fasttext_model.keys())
    # vectors of shape (vocab_size, embedding_dim)
    word_vectors = np.array(list(fasttext_model.values())) 

    # normalize all vectors and convert to unit vectors
    vector_norm = vector / np.linalg.norm(vector)
    word_vectors_norm = word_vectors / np.linalg.norm(word_vectors, axis=1, keepdims=True)

    # cosine similarity with target word
    similarities = np.dot(word_vectors_norm, vector_norm)

    # filter top n words (descending order using partial sorting)
    top_indices = np.argpartition(-similarities, topn)[:topn]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]

    similar_words = []
    for idx in top_indices:
        similar_words.append(vocab[idx])

    return similar_words

def get_unit_vectors(word_vectors):
    word_vectors_norm = word_vectors / np.linalg.norm(word_vectors, axis=1, keepdims=True)
    return word_vectors_norm