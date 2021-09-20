import unicodedata
import numpy as np

class KGRetriever(object):
    def __int__(self):
        self.filepath = ""
        self.max_concept_length = 0
        self.name = "general_kg_retriever"
        self.concept_embedding_mat = [[]]

    def to_dict(self):
        output = dict()
        output['name'] = self.__dict__['name']
        output['max_concept_length'] = self.__dict__['max_concept_length']
        output['concept_vocab_size'] = self.get_concept_vocab_size()
        output['concept_embed_size'] = self.get_concept_embed_size()
        output['file_path'] = self.__dict__['filepath']
        return output

    def get_embedding_mat(self):
        return self.concept_embedding_mat
    def get_concept_embed_size(self):
        return self.concept_embedding_mat.shape[1]
    def get_concept_vocab_size(self):
        return self.concept_embedding_mat.shape[0]
    def get_concept_max_length(self):
        return self.max_concept_length
    def update_max_concept_length(self, num):
        self.max_concept_length = max(self.max_concept_length, num)

    def lookup_concept_ids(self, tokenization_info, **kwargs):
        raise NotImplementedError

    def id2concept_check(self, entity_id):
        return self.id2concept[entity_id]

def read_concept_embedding(embedding_path):
    fin = open(embedding_path, encoding='utf-8')
    info = [line.strip() for line in fin]
    dim = len(info[0].split(' ')[1:])
    n_concept = len(info)
    embedding_mat = []
    id2concept, concept2id = [], {}
    # add padding concept into vocab
    id2concept.append('<pad_concept>')
    concept2id['<pad_concept>'] = 0
    embedding_mat.append([0.0 for _ in range(dim)])
    for line in info:
        concept_name = line.split(' ')[0]
        embedding = [float(value_str) for value_str in line.split(' ')[1:]]
        assert len(embedding) == dim and not np.any(np.isnan(embedding))
        embedding_mat.append(embedding)
        concept2id[concept_name] = len(id2concept)
        id2concept.append(concept_name)
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    fin.close()
    return id2concept, concept2id, embedding_mat

def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)