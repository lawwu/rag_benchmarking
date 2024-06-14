# Modification of bclavie's great script: https://gist.github.com/bclavie/f7b041328615d52cf5c0a9caaf03fd5e
# shared during the Hamel's LLM Conference

import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry
from lancedb.rerankers import CohereReranker

# Fetch some text content in two different categories
from wikipediaapi import Wikipedia
wiki = Wikipedia('RAGBot/0.0', 'en')
docs = [{"text": x,
         "category": "person"}
        for x in wiki.page('Hayao_Miyazaki').text.split('\n\n')]
docs += [{"text": x,
         "category": "film"}
         for x in wiki.page('Spirited_Away').text.split('\n\n')]

# Enter LanceDB
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# Initialise the embedding model
model_registry = get_registry().get("sentence-transformers")
model = model_registry.create(name="BAAI/bge-small-en-v1.5")

# Create a Model to store attributes for filtering
class Document(LanceModel):
    text: str = model.SourceField()
    vector: Vector(384) = model.VectorField()
    category: str

db = lancedb.connect(".my_db")
tbl = db.create_table("my_table", schema=Document)

# Embed the documents and store them in the database
tbl.add(docs)

# Generate the full-text (tf-idf) search index
tbl.create_fts_index("text")

# Initialise a reranker -- here, Cohere's API one
# reranker = CohereReranker()

query = "What is Chihiro's new name given to her by the witch?"

results = (tbl.search(query, query_type="hybrid") # Hybrid means text + vector
.where("category = 'film'", prefilter=True) # Restrict to only docs in the 'film' category
.limit(10) # Get 10 results from first-pass retrieval
# .rerank(reranker=reranker) # For the reranker to compute the final ranking
          )

df_results = results.to_pandas()

print(df_results)

# 0  Plot\nTen-year-old Chihiro Ogino and her paren...  [-0.027931793, 0.019138113, -0.037934814, 0.03...     film          1.000000
# 1  Themes\nSupernaturalism\nThe major themes of S...  [-0.01263991, -0.012689288, -0.060540427, 0.00...     film          0.402163
# 2  Stage "Spirited Away" (Chihiro role: Kanna Has...  [-0.039504554, -0.040483218, 0.06785909, -0.04...     film          0.385661
# 3  Traditional Japanese culture\nSpirited Away co...  [-0.0054386444, 0.051189456, 0.00049261906, -0...     film          0.288939
# 4  Fantasy\nThe film has been compared to Lewis C...  [0.026491504, 0.005764672, 0.008504525, 0.0339...     film          0.253489
# 5  Stage adaptation\nA stage adaptation of Spirit...  [-0.055777255, -0.05455917, 0.059581134, -0.00...     film          0.236336
# 6  Spirited Away (Japanese: 千と千尋の神隠し, Hepburn: Se...  [-0.027961232, -0.02790938, -0.004754297, 0.01...     film          0.221776
# 7  Western consumerism\nSimilar to the Japanese c...  [-0.0036551766, 0.060560934, 0.0022575434, 0.0...     film          0.210290
# 8  Environmentalism\nCommentators have often refe...  [-0.0249137, -0.0074914633, -0.018593505, 0.03...     film          0.142667
# 9  Music\nThe film score of Spirited Away was com...  [-0.049314227, -0.015812704, 0.0023815625, -0....     film          0.026956