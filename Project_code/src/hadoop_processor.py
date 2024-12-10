from pyspark.sql import SparkSession
import subprocess
import os
from pyspark.sql.types import StructType, StructField, StringType, IntegerType,LongType,TimestampType
from pyspark.sql.functions import col
from pyspark.sql.functions import explode

def ensure_hdfs_directories():
    """Ensure HDFS directories exist"""
    commands = [
        'hdfs dfs -mkdir -p /user/landslide/raw_data',
        'hdfs dfs -chmod -R 777 /user/landslide'
    ]
    
    for cmd in commands:
        subprocess.run(cmd.split(), check=True)

def init_hadoop_processing(input_json):
    # To make sure directories exist
    ensure_hdfs_directories()

    # Defining the schema based on expected JSON structure
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("event_title", StringType(), False),
        StructField("event_date", TimestampType(), False),
        StructField("location", StringType(), False),
        StructField("description", StringType(), False),
        StructField("fatalities", StringType(), False),
        StructField("trigger", StringType(), False),
        StructField("country", StringType(), False),
        StructField("url", StringType(), False),
        StructField("score", LongType(), False),
        StructField("num_comments", LongType(), False),
        StructField("matched_keyword", StringType(), False),
        StructField("subreddit", StringType(), False)
    ])
    
    spark = SparkSession.builder \
        .appName("LandslideAnalysis") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .getOrCreate()
    
    try:
        # Read JSON data
        print("Reading JSON data...")
        
        # df = spark.read.schema(schema).option("mode", "PERMISSIVE").json(input_json)
        df = spark.read.option("multiline", "true").json(input_json)
        df.show(truncate=False)

        raw_data = df.rdd.take(5)
        for record in raw_data:
            print(record)

        # df = spark.read.option("mode", "PERMISSIVE").json(input_json)
        df.printSchema()
        
        df.show(truncate=False)

        # Print initial data count
        record_count = df.count()
        print(f"Number of records read from JSON: {record_count}")
        
        if record_count > 0:
            # Save to HDFS
            print("Saving to HDFS...")
            df.write \
                .mode("overwrite") \
                .parquet("hdfs:///user/landslide/raw_data")
            
            print("Data successfully saved to Hadoop")
        else:
            print("No records found in input JSON file")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        
    finally:
        spark.stop()