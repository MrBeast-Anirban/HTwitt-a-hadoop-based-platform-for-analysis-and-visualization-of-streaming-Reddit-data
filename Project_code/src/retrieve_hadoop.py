from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

def retrieve_from_hadoop(local_output_path):
    """
    Retrieve data from Hadoop and append as JSON to local filesystem
    
    Parameters:
    local_output_path (str): Local path where the JSON file should be saved
    
    Returns:
    str: Path where the data was saved
    """
    spark = SparkSession.builder \
        .appName("LandslideDataRetrieval") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .getOrCreate()
    
    try:
        # Read from HDFS Parquet files
        print("Reading data from Hadoop...")
        df = spark.read.parquet("hdfs:///user/landslide/raw_data")
        
        # Print schema and sample data for verification
        print("Schema of retrieved data:")
        df.printSchema()
        
        print("\nSample of retrieved data:")
        df.show(5, truncate=False)
        
        # Get count of records
        record_count = df.count()
        print(f"\nNumber of records retrieved: {record_count}")
        
        if record_count > 0:
            # Create directory if it doesn't exist
            os.makedirs(local_output_path, exist_ok=True)
            
            # Save as JSON to local filesystem using append mode
            print(f"Appending data to local JSON file at {local_output_path}...")
            
            df.coalesce(1) \
              .write \
              .mode("append") \
              .option("multiline", "true") \
              .json(f"file://{local_output_path}")
            
            print("Data successfully appended to local JSON file")
            return local_output_path
        else:
            print("No records found in Hadoop")
            return None
            
    except Exception as e:
        print(f"Error retrieving data: {str(e)}")
        return None
        
    finally:
        spark.stop()


