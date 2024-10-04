from airflow.decorators import dag, task
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from pendulum import datetime, duration
from tabulate import tabulate
import duckdb
import logging
import os

from include.custom_functions.embedding_func import get_embeddings_one_word

t_log = logging.getLogger("airflow.task")

_DUCKDB_INSTANCE_NAME = os.getenv("DUCKDB_INSTANCE_NAME", "include/astronomy.db")
_DUCKDB_TABLE_NAME = os.getenv("DUCKDB_TABLE_NAME", "embeddings_table")
_WORD_OF_INTEREST_PARAMETER_NAME = os.getenv(
    "WORD_OF_INTEREST_PARAMETER_NAME", "my_word_of_interest"
)
_WORD_OF_INTEREST_DEFAULT = os.getenv("WORD_OF_INTEREST_DEFAULT", "star")
_LIST_OF_WORDS_PARAMETER_NAME = os.getenv(
    "LIST_OF_WORDS_PARAMETER_NAME", "my_list_of_words"
)
_LIST_OF_WORDS_DEFAULT = ["sun", "rocket", "planet", "light", "happiness"]


@dag(
    start_date=datetime(2024, 5, 1),
    schedule="@daily",
    catchup=False,
    max_consecutive_failed_dag_runs=5,
    doc_md=__doc__,
    default_args={
        "owner": "Astro",
        "retries": 3,
        "retry_delay": duration(seconds=30),
    },
    tags=["example", "GenAI"],
    params={
        _WORD_OF_INTEREST_PARAMETER_NAME: Param(
            _WORD_OF_INTEREST_DEFAULT,
            type="string",
            title="The word you want to search a close match for.",
            minLength=1,
            maxLength=50,
        ),
        _LIST_OF_WORDS_PARAMETER_NAME: Param(
            _LIST_OF_WORDS_DEFAULT,
            type="array",
            title="A list of words to compare to the word of interest.",
        ),
    },
    max_active_runs=1,
    concurrency=1,
    is_paused_upon_creation=False,
)
def example_vector_embeddings():

    @task(retries=2)
    def get_words(
        **context,
    ) -> list:
        words = context["params"][_LIST_OF_WORDS_PARAMETER_NAME]

        return words

    @task
    def create_embeddings(list_of_words: list) -> list:
        list_of_words_and_embeddings = []

        for word in list_of_words:
            word_and_embeddings = get_embeddings_one_word(word)
            list_of_words_and_embeddings.append(word_and_embeddings)

        return list_of_words_and_embeddings

    @task
    def create_vector_table(
        duckdb_instance_name: str = _DUCKDB_INSTANCE_NAME,
        table_name: str = _DUCKDB_TABLE_NAME,
    ) -> None:
        cursor = duckdb.connect(duckdb_instance_name)

        cursor.execute("INSTALL vss;")
        cursor.execute("LOAD vss;")
        cursor.execute("SET hnsw_enable_experimental_persistence = true;")

        table_name = "embeddings_table"

        cursor.execute(
            f"""
            CREATE OR REPLACE TABLE {table_name} (
                text STRING,
                vec FLOAT[384]
            );

            -- Create an HNSW index on the embedding vector
            CREATE INDEX my_hnsw_index ON {table_name} USING HNSW (vec);
            """
        )
        cursor.close()

    @task
    def insert_words_into_db(
        duckdb_instance_name: str = _DUCKDB_INSTANCE_NAME,
        table_name: str = _DUCKDB_TABLE_NAME,
        list_of_words_and_embeddings: list = None,
    ) -> None:
        cursor = duckdb.connect(duckdb_instance_name)
        cursor.execute("INSTALL vss;")
        cursor.execute("LOAD vss;")

        for i in list_of_words_and_embeddings:
            word = list(i.keys())[0]
            vec = i[word]
            cursor.execute(
                f"""
                INSERT INTO {table_name} (text, vec)
                VALUES (?, ?);
                """,
                (word, vec),
            )

        cursor.close()

    @task
    def embed_word(**context):
        my_word_of_interest = context["params"][_WORD_OF_INTEREST_PARAMETER_NAME]
        embeddings = get_embeddings_one_word(my_word_of_interest)

        embeddings = embeddings[my_word_of_interest]

        return {my_word_of_interest: embeddings}

    @task
    def find_closest_word_match(
        duckdb_instance_name: str = _DUCKDB_INSTANCE_NAME,
        table_name: str = _DUCKDB_TABLE_NAME,
        word_of_interest_embedding: dict = None,
    ):
        cursor = duckdb.connect(duckdb_instance_name)
        cursor.execute("LOAD vss;")

        word = list(word_of_interest_embedding.keys())[0]
        vec = word_of_interest_embedding[word]

        top_3 = cursor.execute(
            f"""
            SELECT text FROM {table_name}
            ORDER BY array_distance(vec, {vec}::FLOAT[384])
            LIMIT 3;
            """
        )

        top_3 = top_3.fetchall()

        t_log.info(f"Top 3 closest words to '{word}':")
        t_log.info(tabulate(top_3, headers=["Word"], tablefmt="pretty"))

        return top_3

    create_embeddings_obj = create_embeddings(list_of_words=get_words())
    embed_word_obj = embed_word()

    chain(
        create_vector_table(),
        insert_words_into_db(list_of_words_and_embeddings=create_embeddings_obj),
        find_closest_word_match(word_of_interest_embedding=embed_word_obj),
    )


example_vector_embeddings()
