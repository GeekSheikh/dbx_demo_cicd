from spark_env import spark

my_df = spark.table("main.tomes_gen.tmdb_movies_bronze").limit(5)

my_df.show()